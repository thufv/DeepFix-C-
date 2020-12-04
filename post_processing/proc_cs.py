import json
import math
import re
import sys

import numpy as np
import subprocess32 as subprocess
import tensorflow as tf
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from data_processing.training_data_generator_cs import vectorize
from neural_net.train import load_data, seq2seq_model
from post_processing.postprocessing_helpers import devectorize, meets_criterion
from util.cs_tokenizer import CS_Tokenizer
from util.helpers import apply_fix, tokens_to_source, vstack_with_right_padding


class FixProgress:

    def __init__(self, raw_code, raw_error_count, tokenized_code,
                 tokenized_code_2, name_dict, error_count, iteration_count):
        # type: (str, int, str, str, Dict[str, str], int, int) -> None
        self.raw_code = raw_code
        self.raw_error_count = raw_error_count
        self.tokenized_code = tokenized_code
        self.tokenized_code_2 = tokenized_code_2
        self.name_dict = name_dict
        self.error_count = error_count
        self.iteration_count = iteration_count

    @staticmethod
    def get_error_count(code):
        # type: (str) -> int
        from subprocess32 import PIPE
        with open('a.cs', 'w') as f:
            f.write(code)
        compilation_result = subprocess.run(
            ['mcs', 'a.cs'], stdout=PIPE, stderr=PIPE)
        if compilation_result.returncode == 0:
            count = 0
        else:
            count = len(re.findall(r'\): error', compilation_result.stderr))
        Path('a.cs').unlink()
        if Path('a.exe').exists():
            Path('a.exe').unlink()
        return count

    @staticmethod
    def from_code(code):
        # type: (str) -> Union[str, FixProgress]
        error_count = FixProgress.get_error_count(code)
        if error_count == 0:
            return code
        tokenized_code, name_dict, _ = CS_Tokenizer().tokenize(code)
        return FixProgress(
            raw_code=code, raw_error_count=error_count,
            tokenized_code=tokenized_code, tokenized_code_2=tokenized_code,
            name_dict=name_dict, error_count=error_count, iteration_count=0)


class FixResult:

    def __init__(self, raw_code, raw_error_count, final_code,
                 final_error_count, iteration_count):
        # type: (str, int, str, int, int) -> None
        self.raw_code = raw_code
        self.raw_error_count = raw_error_count
        self.final_code = final_code
        self.final_error_count = final_error_count
        self.iteration_count = iteration_count

    @staticmethod
    def from_correct_code(code):
        # type: (str) -> FixResult
        return FixResult(raw_code=code, raw_error_count=0, final_code=code,
                         final_error_count=0, iteration_count=0)

    @staticmethod
    def from_final_progress(progress):
        # type: (FixProgress) -> FixResult
        raw_code = progress.raw_code
        raw_error_count = progress.raw_error_count
        final_tokenized_code = progress.tokenized_code_2
        final_code = tokens_to_source(final_tokenized_code, progress.name_dict)
        final_error_count = progress.error_count
        iteration_count = progress.iteration_count
        return FixResult(
            raw_code=raw_code, raw_error_count=raw_error_count,
            final_code=final_code, final_error_count=final_error_count,
            iteration_count=iteration_count)

    def as_dict(self):
        # type: () -> Dict
        return {
            'raw_code': self.raw_code,
            'raw_error_count': self.raw_error_count,
            'final_code': self.final_code,
            'final_error_count': self.final_error_count,
            'iteration_count': self.iteration_count,
        }


class MachineWithSingleNetwork:

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
            configuration=configuration, dataset=dataset,
            raw_model=raw_model, tf_session=session)

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
        # type: (Iterable[str]) -> List[FixResult]
        sequence_of_fix_status = [FixProgress.from_code(code)
                                  for code in sequence_of_code]
        needed_to_fix = [fix_status for fix_status in sequence_of_fix_status
                         if isinstance(fix_status, FixProgress)]
        attempt_count = 0
        while needed_to_fix and attempt_count < 5:
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
            fixes = [devectorize(vector, self.get_dictionary()) for vector in
                     self._get_fixes_ported_from_initial(vectors)]
            for i, fix_progress, fix in zip(range(len(needed_to_fix)),
                                            needed_to_fix, fixes):
                try:
                    tokenized_fixed = apply_fix(
                        fix_progress.tokenized_code,
                        fix, self.get_fix_kind(), flag_replace_ids=False)
                    tokenized_fixed_2 = apply_fix(
                        fix_progress.tokenized_code_2,
                        fix, self.get_fix_kind())
                except Exception:
                    indices_unneeded_to_fix.append(i)
                    continue
                if self.get_task() != 'typo':
                    raise NotImplementedError
                if not meets_criterion(fix_progress.tokenized_code,
                                       fix, 'replace'):
                    indices_unneeded_to_fix.append(i)
                    continue
                error_count_new = FixProgress.get_error_count(tokens_to_source(
                    tokenized_fixed_2, fix_progress.name_dict, False))
                if error_count_new > fix_progress.error_count:
                    indices_unneeded_to_fix.append(i)
                    continue
                fix_progress.tokenized_code = tokenized_fixed
                fix_progress.tokenized_code_2 = tokenized_fixed_2
                fix_progress.error_count = error_count_new
                fix_progress.iteration_count += 1
            for i in reversed(indices_unneeded_to_fix):
                del needed_to_fix[i]
            attempt_count += 1
        results = []
        for fix_status in sequence_of_fix_status:
            if isinstance(fix_status, str):
                results.append(FixResult.from_correct_code(fix_status))
            else:
                results.append(FixResult.from_final_progress(fix_status))
        return results


def get_code_paths_with_pieces_of_code(root):
    # type: (Path) -> List[Tuple[Path, str]]
    results = []
    for code_path in root.glob('**/*.cs'):  # type: Path
        with open(str(code_path)) as f:
            code = f.read()
            if code.startswith('\xef\xbb\xbf'):
                code = code[3:]
            results.append((code_path, code))
    return results


def into_json(code_paths_with_fix_results):
    # type: (Iterable[Tuple[Path, FixResult]]) -> str
    return json.dumps({str(path): result.as_dict() for path, result
                       in code_paths_with_fix_results}, indent=4)


def main():
    # type: () -> None
    code_paths_with_pieces_of_code =\
        get_code_paths_with_pieces_of_code(Path(sys.argv[1]))
    checkpoint_path = Path('data/checkpoints/iitk-typo-1189/bin_0/')
    machine =\
        MachineWithSingleNetwork.from_checkpoint_directory(checkpoint_path)
    print(into_json(zip(
        (path for path, _ in code_paths_with_pieces_of_code),
        machine.process_many(code for _, code in
                             code_paths_with_pieces_of_code))))


if __name__ == '__main__':
    main()
