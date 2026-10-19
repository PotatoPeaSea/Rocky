"""Minimal KiCad s-expression reader/writer used by the Rocky board generators."""
import re

_tok = re.compile(r'\s*(\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+)')


class Sym(str):
    """Bare (unquoted) atom."""


def parse(text):
    stack, cur = [], []
    pos = 0
    n = len(text)
    while pos < n:
        m = _tok.match(text, pos)
        if not m:
            if text[pos:].strip() == '':
                break
            raise ValueError('parse error at %d' % pos)
        t = m.group(1)
        pos = m.end()
        if t == '(':
            stack.append(cur); cur = []
        elif t == ')':
            done = cur; cur = stack.pop(); cur.append(done)
        elif t.startswith('"'):
            cur.append(bytes(t[1:-1], 'utf-8').decode('unicode_escape') if '\\' in t else t[1:-1])
        else:
            cur.append(Sym(t))
    return cur[0] if len(cur) == 1 else cur


def q(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def dump(e, ind=0):
    if isinstance(e, list):
        if not e:
            return '()'
        simple = all(not isinstance(x, list) for x in e)
        if simple:
            return '(' + ' '.join(dump(x) for x in e) + ')'
        out = '(' + dump(e[0])
        for x in e[1:]:
            if isinstance(x, list):
                out += '\n' + '\t' * (ind + 1) + dump(x, ind + 1)
            else:
                out += ' ' + dump(x)
        return out + '\n' + '\t' * ind + ')'
    if isinstance(e, Sym):
        return str(e)
    if isinstance(e, bool):
        return 'yes' if e else 'no'
    if isinstance(e, (int,)):
        return str(e)
    if isinstance(e, float):
        s = ('%.4f' % e).rstrip('0').rstrip('.')
        return s if s not in ('-0', '') else '0'
    return q(e)


def find(e, key):
    return [x for x in e if isinstance(x, list) and x and x[0] == key]


def find1(e, key):
    r = find(e, key)
    return r[0] if r else None


def walk(e):
    yield e
    for x in e:
        if isinstance(x, list):
            yield from walk(x)
