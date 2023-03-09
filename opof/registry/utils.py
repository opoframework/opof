import importlib
import inspect
import os
import re
from multiprocessing import cpu_count
from typing import Callable, List, Optional, Type

from .. import Domain


def concurrency() -> int:
    if "OPOF_CONCURRENCY" in os.environ:
        return int(os.environ["OPOF_CONCURRENCY"])
    else:
        return cpu_count()


def strip_spaces(s: str) -> str:
    return s.replace(" ", "").replace("\t", "")


def to_canonical(args: str) -> str:
    return strip_spaces(args).replace(",", ", ")


def get_signature(c: Callable) -> str:
    s = inspect.signature(c)
    s_formatted = []
    for k in s.parameters.keys():
        default = s.parameters[k].default
        if default == inspect.Parameter.empty:
            s_formatted.append(k)
        else:
            s_formatted.append(k + "=" + str(default))
    return to_canonical(
        f"{c.__name__}[{', '.join(s_formatted)}]".replace("'", "").replace('"', "")
    )


def break_list(s: str) -> List[str]:
    s = strip_spaces(s)
    if len(s) == 0:
        raise ValueError("Empty string")
    if s.startswith(","):
        raise ValueError("List cannot start with delimiter")
    if s.endswith(","):
        raise ValueError("List cannot end with delimiter")

    tokens = []
    token = ""
    nest = 0
    for c in strip_spaces(s):
        if nest == 0 and c == ",":
            if token == "":
                raise ValueError("Empty token")
            tokens.append(token)
            token = ""
        elif c == "[":
            nest += 1
            token += c
        elif c == "]":
            if nest == 0:
                raise ValueError("Unmatched brace")
            nest -= 1
            token += c
        else:
            token += c
    if token != "":
        tokens.append(token)
    return tokens


def to_eval_string(s: str):
    s = strip_spaces(s)
    if len(s) == 0:
        raise ValueError("Empty string")

    # Check list type.
    if s.startswith("[") and s.endswith("]"):
        return "[" + ",".join([to_eval_string(t) for t in break_list(s[1:-1])]) + "]"

    # Try to parse..
    try:
        import torch
        import numpy as np
        eval(s)
        return s
    except:
        pass

    # Assume string.
    return f"'{s}'"


def parse_callable(s: str):
    for t in ['"', "'"]:
        if t in s:
            raise ValueError("Invalid callable")

    r = re.search(r"^([^\d\W]\w*)\[(.*)\]\Z", s)
    if r is None:
        raise ValueError("Invalid callable")

    (f, args) = r.groups()
    args_list = break_list(args)
    args_parsed = []
    for a in args_list:
        if re.match(r"^([^\d\W]\w*)=(.*)\Z", a):
            (k, v) = a.split("=", 1)
            v = to_eval_string(v)
            args_parsed.append(f"{k}={v}")
        else:
            args_parsed.append(to_eval_string(a))

    return (f, ",".join(a for a in args_parsed).replace(",", ", "))


def get_domain_classes() -> List[Type[Domain]]:
    if "OPOF_DOMAINS" not in os.environ:
        return []

    domain_classes = []
    for module_name in os.environ["OPOF_DOMAINS"].split(";"):
        module = importlib.import_module(module_name)
        for _, m in inspect.getmembers(module):
            if inspect.isclass(m):
                if issubclass(m, Domain):
                    domain_classes.append(m)

    return domain_classes


def get_algorithms() -> List[Type]:
    module = importlib.import_module("opof.algorithms")
    algorithm_classes = []
    for (_, m) in inspect.getmembers(module):
        if inspect.isclass(m):
            algorithm_classes.append(m)
    return algorithm_classes


def resolve_domain_class(name: str) -> Optional[Type[Domain]]:
    # Resolve algorithm.
    domain_class = None
    for d in get_domain_classes():
        if d.__name__ == name:
            if domain_class is not None:
                raise Exception("Multiple domain class matches")
            domain_class = d
    return domain_class


def resolve_algorithm(name: str) -> Optional[Type]:
    # Resolve algorithm.
    algorithm = None
    for a in get_algorithms():
        if a.__name__ == name:
            if algorithm is not None:
                raise Exception("Multiple algorithm matches")
            algorithm = a
    return algorithm
