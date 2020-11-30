import re
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from neural_net.train import load_data, seq2seq_model


class MachineWithSingleNetwork:

    class ProcessResult:

        def __init__(self):
            # type: () -> None
            pass

    def __init__(self, dataset, raw_model, tf_session):
        # type: (load_data, seq2seq_model, tf.Session) -> None
        self.dataset = dataset
        self.raw_model = raw_model
        self.tf_session = tf_session

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
        configuration = np.load(
            path/'experiment-configuration.npy',
            allow_pickle=True).item()  # type: Any
        data_directory = configuration['args'].data_directory  # type: str
        dataset = load_data(
            data_directory, shuffle=False, load_only_dicts=True)
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
            dataset=dataset, raw_model=raw_model, tf_session=session)

    def process(self, code):
        # type: (str) -> ProcessResult
        pass


def main():
    # type: () -> None
    checkpoint_path = Path('data/checkpoints/iitk-typo-1189/bin_0/')
    machine = MachineWithSingleNetwork.from_checkpoint_directory(checkpoint_path)


if __name__ == '__main__':
    main()
