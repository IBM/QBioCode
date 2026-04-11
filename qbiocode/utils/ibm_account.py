# This will be a simple function to extract information from a user's qiskit-json file

import json
import os

from qiskit_ibm_runtime import QiskitRuntimeService


def get_creds(args):
    """This function determines the user's IBM Quantum channel, instance, and token, using values provided
    within the config.yaml file or as defined within the user's qiskit configuration from provided qiskit_json_path
    specified in the config.yaml file, and then parses its contents. It returns the main items in this json file,
    such as the instance and api token, which can then be passed into the QML functions when using a real
    hardware backend.
    The function will return a dictionary with the keys 'channel', 'instance', 'token', and 'url',
    which can be used to instantiate the QiskitRuntimeService.
    If the qiskit_json_path is provided, it will attempt to read the credentials from that file.
    Args:
        args (dict): This passes the arguments from the config.yaml file.  In this particular case, it is importing the path to the qiskit-ibm.json file (qiskit_json_path) and the credentials
        defined in this json file (ibm_channel, ibm_instance, ibm_token, ibm_url).

    Returns:
        rval (dict): A dictionary containing the IBM Quantum credentials, including 'channel', 'instance', 'token', and 'url'.
    """
    cred_source_dict = {
        "channel": "ibm_channel",
        "instance": "ibm_instance",
        "token": "ibm_token",
        "url": "ibm_url",
    }
    rval = {}
    for ibm_name, yaml_name in cred_source_dict.items():
        value = args.get(yaml_name, None)
        if value:
            rval[ibm_name] = value

    qiskit_json_path = args.get("qiskit_json_path", None)
    if qiskit_json_path:
        qiskit_json_path = os.path.expanduser(qiskit_json_path)
        if os.path.exists(qiskit_json_path):
            # load the qiskit json file
            with open(qiskit_json_path, "r") as jfile:
                creds = json.load(jfile)
            # Access keys and values
            # The items we want are actually in a nested dictionary, so we have to loop through the outer dictionary first, then the
            # nested one.  This nested dictionary (outer_value) is actually the value for the key in the parent dictionary.
            for outer_key, outer_value in creds.items():
                if "name" in rval.keys() and outer_key == rval["name"]:
                    for ibm_name in cred_source_dict.keys():
                        if ibm_name not in rval:
                            value = outer_value.get(ibm_name, None)
                            if value:
                                rval[ibm_name] = value
        else:
            print(
                "IBM credentials not found! Please verify that the path to your qiskit-ibm.json file is correct."
            )
    return rval


def instantiate_runtime_service(args):
    """This function provides a quick way to instantiate QiskitRuntimeService in one place. A basic call to this function can then be done in anywhere else.
    It uses the get_creds function to retrieve the necessary credentials from the qiskit-ibm.json file, with the file path specified in the config.yaml file.
    It returns an instance of the QiskitRuntimeService class, which can be used to interact with IBM Quantum services.

    Args:
        args (dict): This passes the arguments from the config.yaml file.  In this particular case, it is importing the path to the qiskit-ibm.json file (qiskit_json_path) and the credentials
        defined in this json file (ibm_channel, ibm_instance, ibm_token, ibm_url).

    Returns:
        QiskitRuntimeService: An instance of the QiskitRuntimeService class, initialized with the credentials from the qiskit-ibm.json file or the provided arguments.
    """
    return QiskitRuntimeService(**get_creds(args))
