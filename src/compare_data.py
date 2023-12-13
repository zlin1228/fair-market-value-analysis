import os
import sys
from types import SimpleNamespace
from typing import cast, Dict, Iterable, Tuple
import zipfile

import lz4.frame
import pandas as pd

from performance_helper import create_profiler
import s3
from trace_data import load_h5, load_parquet

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

def _format_strings(l: Iterable[str]):
    s = list(l)
    s.sort()
    return ', '.join(s)

def load(path: str) -> Tuple[pd.DataFrame,Dict[str,pd.Series]]:
    if path.startswith('s3://'):
        # s3 file
        file_list = s3.get_object(path)
        if file_list[0].endswith('.h5'):
            return load_h5(file_list[0])
        elif file_list[0].endswith('.parquet'):
            return load_parquet(file_list)
        else:
            raise NotImplementedError(f'data file extension not recognized: {file_list}')
    else:
        # local file
        if path.endswith('.h5'):
            return load_h5(path)
        elif path.endswith('.lz4'):
            decompressed_path = os.path.splitext(path)[0]
            with create_profiler(f'decompress {path} to {decompressed_path}')():
                with lz4.frame.open(path, mode='rb') as lz4_file:
                    with open(decompressed_path, 'wb') as output_file:
                        output_file.write(cast(bytes, lz4_file.read()))
            return load_h5(decompressed_path)
        elif path.endswith('.zip'):
            expanded_path = os.path.splitext(path)[0]
            with create_profiler(f'decompress {path} to {expanded_path}')():
                with zipfile.ZipFile(path, 'r') as zf:
                    namelist = zf.namelist()
                    zf.extractall(expanded_path)
            return load_parquet([os.path.join(expanded_path, file) for file in namelist if not file.endswith('/')])
        else:
            raise NotImplementedError(f'data file extension not recognized: {path}')

def apply_ordinals(data: Tuple[pd.DataFrame,Dict[str,pd.Series]]):
    trace, ordinals = data
    profile = create_profiler('apply ordinals')
    with profile():
        for c in ordinals:
            with profile(f'apply {c}'):
                r_o = { v: k for k, v in ordinals[c].items() }
                trace[c] = trace[c].map(r_o)
        return trace, ordinals

def compare_data(d1: Tuple[pd.DataFrame,Dict[str,pd.Series]], d2: Tuple[pd.DataFrame,Dict[str,pd.Series]], s=None) -> bool:
    profile = create_profiler('compare data')
    equivalent: bool = True
    with profile() as p:
        df1, _ = d1
        df2, _ = d2
        if df1.shape[0] == df2.shape[0]:
            p.print(f'same row count: {df1.shape[0]}')
        else:
            p.print('DIFFERENT ROW COUNTS')
            print(df1.shape[0])
            print(df2.shape[0])
            equivalent: bool = False

        if set(df1.columns.to_list()) == set(df2.columns.to_list()):
            p.print(f'same columns: {_format_strings(df1.columns.to_list())}')
        else:
            p.print('DIFFERENT COLUMNS')
            print(_format_strings(df1.columns.to_list()))
            print(_format_strings(df2.columns.to_list()))
            equivalent: bool = False

        # default to printing out 10 evenly spaced values
        if s is None:
            s = slice(0, None, int(df1.shape[0] / 10) or 1)
        print_range = range(*s.indices(df1.shape[0]))
        total_differences: int = 0
        for c in df1.columns:
            with create_profiler(c)() as cp:
                differences: int = 0
                for i, (v1, v2) in enumerate(zip(df1[c], df2[c])):
                    if i in print_range:
                        cp.print(f'{f"{i}:": <10} "{v1}" "{v2}"')
                    if v1 != v2 and (pd.notna(v1) or pd.notna(v2)):
                        equivalent: bool = False
                        differences += 1
                        if differences == 1:
                            cp.print(f'EXAMPLE DIFFERENCE IN COLUMN: {c}, ROW: {i}')
                            print(f'"{v1}"')
                            print(f'"{v2}"')
                cp.print('equivalent' if differences == 0 else f'{differences} differences')
                total_differences += differences
        if equivalent:
            p.print('equivalent')
        elif total_differences == 0:
            p.print('first contains equivalent subset of second')
        else:
            p.print(f'{total_differences} total differences')
        return equivalent

def _compare_ordinal(o1: pd.Series, o2: pd.Series, p, s=None) -> bool:
    equivalent: bool = True
    if o1.shape[0] == o2.shape[0]:
        p.print(f'same row count: {o1.shape[0]}')
    else:
        p.print('DIFFERENT ROW COUNTS')
        print(o1.shape[0])
        print(o2.shape[0])
        equivalent: bool = False

    # default to printing out 10 evenly spaced values
    if s is None:
        s = slice(0, None, int(o1.shape[0] / 10) or 1)
    print_range = range(*s.indices(o1.shape[0]))
    differences: int = 0
    for i, (i1, v1) in enumerate(o1.items()):
        v2 = o2[i1]
        if i in print_range:
            i1s = f'"{i1}"'
            p.print(f'{v1: <6} {v2: <6} {i1s} ')
        if v1 != v2:
            equivalent: bool = False
            differences += 1
            if differences == 1:
                p.print(f'EXAMPLE DIFFERENCE: "{i1}"')
                print(f'"{v1}"')
                print(f'"{v2}"')
    if equivalent:
        p.print('equivalent')
    elif differences == 0:
        p.print('first contains equivalent subset of second')
    else:
        p.print(f'{differences} differences')
    return equivalent

def compare_ordinals(d1: Tuple[pd.DataFrame,Dict[str,pd.Series]], d2: Tuple[pd.DataFrame,Dict[str,pd.Series]], s=None) -> bool:
    profile = create_profiler('compare ordinals')
    with profile() as p:
        _, o1 = d1
        _, o2 = d2
        if set(o1.keys()) == set(o2.keys()):
            p.print(f'same ordinal columns: {_format_strings(o1.keys())}')
        else:
            p.print('DIFFERENT ORDINAL COLUMNS')
            print(_format_strings(o1.keys()))
            print(_format_strings(o2.keys()))
            return False

        result = {}
        for o in o1.keys():
            with create_profiler(f'compare ordinal: {o}')() as op:
                result[o] = _compare_ordinal(o1[o], o2[o], op, s)
        if len(e := [k for k,v in result.items() if v]) > 0:
            p.print(f'equivalent ordinals: {_format_strings(e)}')
        if len(d := [k for k,v in result.items() if not v]) > 0:
            p.print(f'differing ordinals: {_format_strings(d)}')

        return len(d) == 0

# simple namespace to provide easier access to everything when running interactively
n = SimpleNamespace(
    p1 = '',
    p2 = '')
# add some helper functions to namespace
def add_to_namespace():
    """ Example usage:

        # Start Python in interactive mode and load file
        $ python
        >>> from compare_h5 import *

        # load the two dataframes from the default locations
        >>> n.l()

        # print 5 rows from the start, middle, and end of each dataframe
        >>> n.p()

        # apply any ordinals to both dataframes
        >>> n.a()

        # print 10 rows from start, middle, and end of updated dataframes
        >>> n.p(10)

        # compare the two dataframes (prints 10 values from each column during compare)
        >>> n.c()

        # compare the two dataframes printing the first 20 values from each column as it is being compared
        >>> n.c(slice(0, 20))
    """
    def _load_interactive():
        n.d1 = load(n.p1)
        n.d2 = load(n.p2)
    def _apply_ordinals_interactive():
        n.d1 = apply_ordinals(n.d1)
        n.d2 = apply_ordinals(n.d2)
    def _print_interactive(row_count = 5):
        print_full(n.d1[0][0:row_count])
        print_full(n.d1[0][(h1 := int(n.d1[0].shape[0] / 2)): h1 + row_count])
        print_full(n.d1[0][-row_count:])
        print_full(n.d2[0][0:row_count])
        print_full(n.d2[0][(h2 := int(n.d2[0].shape[0] / 2)): h2 + row_count])
        print_full(n.d2[0][-row_count:])
    def _compare_data_interactive(s=None):
        compare_data(n.d1, n.d2, s)
    def _compare_ordinals_interactive(s=None):
        compare_ordinals(n.d1, n.d2, s)
    def _compare_interactive(s=None):
        _compare_data_interactive(s)
        _compare_ordinals_interactive(s)
    n.l = _load_interactive
    n.a = _apply_ordinals_interactive
    n.p = _print_interactive
    n.cd = _compare_data_interactive
    n.co = _compare_ordinals_interactive
    n.c = _compare_interactive
add_to_namespace()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python compare_h5.py <path to first> <path to second>')
        exit()
    p1 = sys.argv[1]
    p2 = sys.argv[2]
    d1 = load(p1)
    d2 = load(p2)
    equivalent_data = compare_data(apply_ordinals(d1), apply_ordinals(d2))
    equivalent_ordinals = compare_ordinals(d1, d2)
    print(f'data is {"" if equivalent_data else "NOT "}equivalent "{p1}"  "{p2}"')
    print(f'ordinals are {"" if equivalent_ordinals else "NOT "}equivalent "{p1}"  "{p2}"')
