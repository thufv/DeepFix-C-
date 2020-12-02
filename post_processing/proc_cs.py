import math
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from pathlib import Path

from data_processing.training_data_generator_cs import vectorize
from neural_net.train import load_data, seq2seq_model
from util.cs_tokenizer import CS_Tokenizer
from util.helpers import apply_fix, tokens_to_source, vstack_with_right_padding
from .postprocessing_helpers import devectorize, meets_criterion


class MachineWithSingleNetwork:

    class FixProgress:

        def __init__(self, tokenized_code, tokenized_code_2,
                     name_dict, error_count):
            # type: (str, str, Dict[str, str], int) -> None
            self.tokenized_code = tokenized_code
            self.tokenized_code_2 = tokenized_code_2
            self.name_dict = name_dict
            self.error_count = error_count

        @staticmethod
        def get_error_count(code):
            # type: (str) -> int
            raise NotImplementedError

        @staticmethod
        def from_code(code):
            # type: (str) -> Union[str, MachineWithSingleNetwork.FixProgress]
            error_count =\
                MachineWithSingleNetwork.FixProgress.get_error_count(code)
            if error_count == 0:
                return code
            tokenized_code, name_dict, _ = CS_Tokenizer().tokenize(code)
            return MachineWithSingleNetwork.FixProgress(
                tokenized_code=tokenized_code, tokenized_code_2=tokenized_code,
                name_dict=name_dict, error_count=error_count)

    def __init__(self, configuration, dataset, raw_model, tf_session):
        # type: (Any, load_data, seq2seq_model, tf.Session) -> None
        self.configuration = configuration
        self.dataset = dataset
        self.raw_model = raw_model
        self.tf_session = tf_session

    def get_dictionary(self):
        # type: () -> load_data
        return self.dataset.get_tl_dictionary()

    def get_task(self):
        # type: () -> str
        return self.configuration['which_network']

    def is_id_dropped(self):
        # type: () -> bool
        return self.get_task() == 'typo'

    def get_fix_kind(self):
        # type: () -> str
        if self.get_task() == 'typo':
            return 'replace'
        return 'insert'

    @staticmethod
    def get_best_checkpoint_identifier(checkpoint_path):
        # type: (Path) -> int
        best = None
        regex = re.compile(r'saved-model-attn-(\d+)\.meta\Z')
        for item_path in (checkpoint_path/'best').iterdir():
            match = regex.match(item_path.name)
            if match is None:
                continue
            identifier = int(match.group(1))
            if best is None or identifier > best:
                best = identifier
        if best is None:
            raise RuntimeError
        return best

    @staticmethod
    def from_checkpoint_directory(path):
        # type: (Path) -> MachineWithSingleNetwork
        configuration = np.load(path/'experiment-configuration.npy',
                                allow_pickle=True).item()  # type: Any
        data_directory = configuration['args'].data_directory  # type: str
        dataset = load_data(data_directory, shuffle=False,
                            load_only_dicts=True)
        with tf.variable_scope(configuration['which_network']):
            raw_model = seq2seq_model(
                dataset.vocabulary_size, 50, 28, cell_type='LSTM',
                memory_dim=300, num_layers=4, dropout=0)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        best = MachineWithSingleNetwork.get_best_checkpoint_identifier(path)
        best = str(path/'best'/'saved-model-attn-{}'.format(best))
        raw_model.load_parameters(session, best)
        return MachineWithSingleNetwork(
            configuration=configuration, dataset=dataset, raw_model=raw_model,
            tf_session=session)

    def vectorize(self, tokenized_code):
        # type: (str) -> Optional[List[int]]
        try:
            return vectorize(tokenized_code, self.get_dictionary(), 450,
                             self.is_id_dropped(), reverse=True)
        except KeyError:
            return

    def _get_fixes_in_batch_ported_from_initial(self, vectors):
        x, x_len = tuple(self.dataset.prepare_batch(vectors))
        fixes = self.raw_model.sample(self.tf_session, x, x_len)
        assert len(vectors) == np.shape(fixes)[0]
        return fixes

    def _get_fixes_ported_from_initial(self, vectors):
        num_programs = len(vectors)
        all_fixes = []
        for i in range(int(math.ceil(num_programs * 1.0 / 100))):
            start = i * 100
            end = (i + 1) * 100
            fixes = self._get_fixes_in_batch_ported_from_initial(
                vectors[start:end])
            all_fixes.append(fixes)
        fixes = vstack_with_right_padding(all_fixes)
        assert num_programs == np.shape(fixes)[0],\
            ('num_programs: {}, fixes-shape: {}'
             .format(num_programs, np.shape(fixes)))
        return fixes

    def process_many(self, sequence_of_code):
        # type: (Iterable[str]) -> List[Tuple[str, int]]
        sequence_of_fix_status = [
            MachineWithSingleNetwork.FixProgress.from_code(code)
            for code in sequence_of_code
        ]  # type: List[Union[str, MachineWithSingleNetwork.FixProgress]]
        needed_to_fix = [
            fix_status for fix_status in sequence_of_fix_status
            if isinstance(fix_status, MachineWithSingleNetwork.FixProgress)
        ]  # type: List[MachineWithSingleNetwork.FixProgress]
        while needed_to_fix:
            indices_unneeded_to_fix = []
            vectors = []
            for i, fix_progress in enumerate(needed_to_fix):
                vector = self.vectorize(fix_progress.tokenized_code)
                if vector is None:
                    indices_unneeded_to_fix.append(i)
                else:
                    vectors.append(vector)
            for i in reversed(indices_unneeded_to_fix):
                del needed_to_fix[i]
            indices_unneeded_to_fix = []
            fixes = [
                devectorize(vector, self.get_dictionary()) for vector in
                self._get_fixes_ported_from_initial(vectors)]
            for i, fix_progress, fix in zip(range(len(needed_to_fix)),
                                            needed_to_fix, fixes):
                try:
                    tokenized_fixed = apply_fix(
                        fix_progress.tokenized_code, fix, self.get_fix_kind(),
                        flag_replace_ids=False)
                    tokenized_fixed_2 = apply_fix(
                        fix_progress.tokenized_code_2, fix, self.get_fix_kind())
                except Exception:
                    indices_unneeded_to_fix.append(i)
                    continue
                if self.get_task() != 'typo':
                    raise NotImplementedError
                if not meets_criterion(fix_progress.tokenized_code, fix, 'replace'):
                    indices_unneeded_to_fix.append(i)
                    continue
                error_count_new = (
                    MachineWithSingleNetwork.FixProgress.get_error_count(
                        tokens_to_source(
                            tokenized_fixed_2, fix_progress.name_dict, False)))
                if error_count_new > fix_progress.error_count:
                    indices_unneeded_to_fix.append(i)
                    continue
                fix_progress.tokenized_code = tokenized_fixed
                fix_progress.tokenized_code_2 = tokenized_fixed_2
                fix_progress.error_count = error_count_new
            for i in reversed(indices_unneeded_to_fix):
                del needed_to_fix[i]
        results = []
        for fix_status in sequence_of_fix_status:
            if isinstance(fix_status, str):
                results.append((fix_status, 0))
            else:
                tokenized_code = fix_status.tokenized_code
                name_dict = fix_status.name_dict
                error_count = fix_status.error_count
                results.append((tokens_to_source(tokenized_code, name_dict),
                                error_count))
        return results


def get_sequence_of_code(root):
    # type: (Path) -> Iterable[str]
    for code_path in root.glob('*/*.cs'):  # type: Path
        with open(str(code_path)) as file:
            yield file.read()


def main():
    # type: () -> None
    checkpoint_path = Path('data/checkpoints/iitk-typo-1189/bin_0/')
    machine =\
        MachineWithSingleNetwork.from_checkpoint_directory(checkpoint_path)


if __name__ == '__main__':
    main()
