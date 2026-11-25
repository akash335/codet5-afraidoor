"""Fixed input-side identifier normalization for the CodeT5 defense evaluation."""
import ast
import hashlib
import io
import token
import tokenize

SKIP_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp,
              ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Global, ast.Nonlocal,
              ast.Import, ast.ImportFrom, ast.ExceptHandler, ast.AnnAssign, ast.JoinedStr)
DYNAMIC_NAMES = {"locals", "globals", "vars", "dir", "eval", "exec", "compile", "inspect",
                 "getframe", "_getframe", "f_locals", "currentframe"}
IGNORED = {tokenize.COMMENT, tokenize.NL, token.NEWLINE, token.INDENT, token.DEDENT, token.ENDMARKER}


def rename_locals(row):
    """Return the model input after a fixed refactor, plus a coverage reason."""
    code = row["source_code"]
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return row, "unparseable_source"
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return row, "outside_single_function"
    function = tree.body[0]
    nodes = [n for stmt in function.body for n in ast.walk(stmt)]
    if any(isinstance(n, SKIP_NODES) or type(n).__name__.startswith("Match") for n in nodes):
        return row, "unsupported_scope_or_construct"
    if any(getattr(n, "id", None) in DYNAMIC_NAMES or getattr(n, "attr", None) in DYNAMIC_NAMES for n in nodes):
        return row, "dynamic_name_access"
    parameters = {n.arg for n in ast.walk(function.args) if isinstance(n, ast.arg)}
    names = [n for n in nodes if isinstance(n, ast.Name)]
    stores = sorted((n for n in names if isinstance(n.ctx, ast.Store)), key=lambda n: (n.lineno, n.col_offset))
    # Preserve public names, private/magic spellings, self, cls and all parameters.
    candidates = dict.fromkeys(n.id for n in stores if n.id not in parameters | {"self", "cls", function.name}
                               and not n.id.startswith("_"))
    if not candidates:
        return row, "no_eligible_local"
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    if [t.string for t in toks if t.type not in IGNORED] != row["code_tokens"]:
        return row, "source_token_mismatch"
    existing = {t.string for t in toks if t.type == token.NAME}
    serial = 0
    mapping = {}
    for name in candidates:
        while True:
            serial += 1
            replacement = f"local_{serial}"
            if replacement not in existing:
                mapping[name] = replacement
                break
    # AST columns are UTF-8 byte offsets. Token positions use Unicode characters.
    lines = code.split("\n")
    positions = {}
    for node in names:
        if node.id in mapping:
            prefix = lines[node.lineno - 1].encode()[:node.col_offset].decode()
            positions[(node.lineno, len(prefix))] = mapping[node.id]
            node.id = mapping[node.id]
    rewritten = [t._replace(string=positions[t.start]) if t.type == token.NAME and t.start in positions else t
                 for t in toks]
    source = tokenize.untokenize(rewritten)
    if ast.dump(ast.parse(source)) != ast.dump(tree):
        return row, "structural_check_failed"
    return {**row, "source_code": source, "code_tokens": [t.string for t in rewritten if t.type not in IGNORED],
            "sha256_hash": hashlib.sha256(source.encode()).hexdigest()}, "renamed"
