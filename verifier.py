import argparse
import onnx
import logging
import time

import pynever.networks as pyn_networks
import pynever.strategies.conversion as pyn_conv
import pynever.strategies.verification as pyn_ver
import utilities as utils


def make_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser("Neural Network Verifier for the AIDOaRt Project")
    parser.add_argument("--model_path", type=str, default="models/VC_FE=[32-64-128-256-512-1024]_C=[64-32].onnx",
                        help="Path to the ONNX file containing the Neural Network to verify")
    parser.add_argument("--property_path", type=str, default="properties/VC_FE=[32-64-128-256-512-1024]"
                                                             "_C=[64-32]_CLS_e=0.1.smtlib",
                        help="Path to the smtlib file containing the property of interest.")
    parser.add_argument("--ver_heuristic", type=str, default="overapprox",
                        help="Verification heuristic to apply to the network and property of interest."
                             "Should be one among 'overapprox', 'mixed', 'complete'.")
    parser.add_argument("--logs_path", type=str, default="logs/ver_logs.csv", help="Filepath to the logs file.")

    return parser


def instantiate_logger(filepath: str):

    logger_path = filepath

    stream_logger = logging.getLogger("pynever.strategies.verification")
    file_logger = logging.getLogger("Log File")

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logger_path, 'a+')

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    stream_logger.addHandler(stream_handler)
    file_logger.addHandler(file_handler)

    stream_logger.setLevel(logging.INFO)
    file_logger.setLevel(logging.INFO)

    return stream_logger, file_logger


if __name__ == "__main__":

    # Parse the command line parameters.
    arg_parser = make_parser()
    args = arg_parser.parse_args()

    model_path = args.model_path
    model_id = model_path.split('/')[-1].replace('.onnx', '')

    property_path = args.property_path
    property_id = property_path.split('/')[-1].replace('.smtlib', '')

    ver_heuristic = args.ver_heuristic
    logs_path = args.logs_path

    # Instantiate loggers.
    stream_log, file_log = instantiate_logger(logs_path)

    # Load the model in the pynever format.
    onnx_net = pyn_conv.ONNXNetwork(model_id, onnx.load(model_path))
    pyn_net = pyn_conv.ONNXConverter().to_neural_network(onnx_net)

    # It is assumed that the input model is a sequential neural network in the form
    # [convolutional feature extractor] -> [flatten layer] -> [fully connected classifier] or directly the fully
    # connected classifier.
    # The property is assumed to be defined on the input and output of the fully-connected classifier.

    if not isinstance(pyn_net, pyn_networks.SequentialNetwork):
        raise NotImplementedError

    # Check if the model contains a convolutional feature extractor
    if utils.is_convolutional(pyn_net):
        # if so, isolate the classifier from the feature extractor
        pyn_net_to_verify = utils.extract_cls(pyn_net)
    else:
        # otherwise keep the network as it is.
        pyn_net_to_verify = pyn_net

    # Now we load the property of interest.
    never_prop = pyn_ver.NeVerProperty()
    never_prop.from_smt_file(property_path)

    stream_log.info(f"Verifying Model {pyn_net_to_verify.identifier} and Property {property_id}...")

    # We prepare the verification strategy with the verification heuristic chosen.
    ver_strategy = pyn_ver.NeverVerification(heuristic=ver_heuristic)

    start = time.perf_counter()
    is_verified = ver_strategy.verify(pyn_net_to_verify, never_prop)
    end = time.perf_counter()

    stream_log.info(f"Verification Result: {is_verified}")
    file_log.info(f"{model_path},{property_path},{is_verified},{end - start}")
