# AIDOaRt-UNISS-NNVer

NNVer is a verification tool designed to enhance the reliability of Neural Networks (NNs) for safety-critical CPSs. 
Our solution introduces "step-zero object detection" networks alongside intricate object detection NNs to simplify 
verification processes. By prioritising object classification over precise localization, NNVer ensures formal 
certification of step-zero models alongside the original object detection model, bolstering safety by alerting to 
object presence even when exact locations are unclear.

## Requirements
NNVer requires [pyNeVer](https://github.com/NeVerTools/pyNeVer) and all its dependencies. We refer to its 
[Github Page](https://github.com/NeVerTools/pyNeVer) for more information regarding how to install them.

## How to use
NNVer can be executed launching [main.py](main.py) and requires four command line parameters:
- `--model_path`: Path to the ONNX file containing the Neural Network to verify.
An example can be found in [test_model.onnx](models/test_model.onnx).
- `--property_path`: Path to the .smt2 file containing the property of interest. 
An example can be found in [test_property.smt2](properties/test_property.smt2).
- `--output_path`: Path to the Plain Text file that will contain the results of the verification process.
An example can be found in [output.csv](outputs/output.csv).
- `--config_path`: Path to the .ini file containing the configuration info for the script.
An example can be found in [default_config.ini](configs/default_config.ini).

## Important Notes

- If new variables are added in the config file, minor modification of the code in [main.py](main.py) will be needed
to manage them. The variable `ver_heuristic` should be one among *overapprox*, *mixed*, and *complete*.