from typing import Dict, Union, List, Tuple

NestedStringDict = Dict[str, Union[str, 'NestedStringDict']]


def toml_dict_to_dotted_strings(toml_dict: NestedStringDict) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []

    for key, value in toml_dict.items():
        prefix = key

        if isinstance(value, dict):
            suffixes = toml_dict_to_dotted_strings(value)

            for suffix in suffixes:
                result.append((prefix + '.' + suffix[0], suffix[1]))
        else:
            result.append((key, value))

    return result
