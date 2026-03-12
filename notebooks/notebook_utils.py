import inspect

def verify_novel(novel_str, novel_name_str=None, index_ends=500):
    """
    INPUT: string in some stage of processing
    OUTPUT: display summary index_ends chars of header/footer for verification

    This utility will attempt to use a provided novel_name_str. If none is provided,
    it tries to read a variable named `novel_name_str` from the caller's globals/locals
    so it works when called from a notebook cell that defines that variable.
    """
    if novel_name_str is None:
        try:
            frame = inspect.currentframe()
            outer = frame.f_back
            if outer is not None:
                novel_name_str = outer.f_globals.get('novel_name_str', outer.f_locals.get('novel_name_str', ''))
            else:
                novel_name_str = ''
        except Exception:
            novel_name_str = ''

    print(f'Novel Name: {novel_name_str}')
    try:
        length = len(novel_str)
    except Exception:
        length = 0
    print(f'  Char Len: {length}')
    print('====================================\\n')
    print(f'Beginning:\\n\\n {str(novel_str)[:index_ends]}\\n\\n')
    print('\\n------------------------------------')
    print(f'Ending:\\n\\n {str(novel_str)[-index_ends:]}\\n\\n')