import pynever.nodes as pyn_nodes
import pynever.networks as pyn_networks
import logging


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


def is_convolutional(pyn_net: pyn_networks.SequentialNetwork) -> bool:

    current_node = pyn_net.get_first_node()
    is_conv = False
    while current_node is not None:

        if isinstance(current_node, pyn_nodes.ConvNode):
            is_conv = True

        current_node = pyn_net.get_next_node(current_node)

    return is_conv


def extract_cls(pyn_net: pyn_networks.SequentialNetwork) -> pyn_networks.SequentialNetwork:

    pyn_cls = pyn_networks.SequentialNetwork(pyn_net.identifier + "_CLS", "X")
    current_node = pyn_net.get_first_node()
    flatten_reached = False
    while current_node is not None:

        # We scroll the network until we reach the flatten layer
        if not flatten_reached:

            if isinstance(current_node, pyn_nodes.FlattenNode):
                flatten_reached = True

        # Then we add all the following nodes to the network to verify
        else:
            pyn_cls.add_node(current_node)

        current_node = pyn_net.get_next_node(current_node)

    return pyn_cls

