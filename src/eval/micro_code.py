"""Tiny original fixed Python checks; interpret arithmetic AST, never exec model text."""
from __future__ import annotations
import ast,hashlib,json,operator
import torch
from tokenizers import Tokenizer
from src.training.engine import amp

TASKS=[('sum','a + b',[(2,3,5),(-3,8,5),(0,0,0)]),('difference','a - b',[(8,3,5),(-3,2,-5),(1,1,0)]),
       ('product','a * b',[(2,4,8),(-2,3,-6),(0,7,0)]),('maximum','a if a > b else b',[(3,4,4),(5,2,5),(-3,-7,-3)]),
       ('minimum','a if a < b else b',[(3,4,3),(5,2,2),(-3,-7,-7)]),('remainder','a % b',[(7,3,1),(10,4,2),(12,5,2)])]
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Mod:operator.mod,ast.FloorDiv:operator.floordiv,ast.Gt:operator.gt,ast.Lt:operator.lt,ast.Eq:operator.eq}
def expression(node,values):
    if isinstance(node,ast.Name) and node.id in values:return values[node.id]
    if isinstance(node,ast.Constant) and type(node.value) in (int,bool) and abs(node.value)<10000:return node.value
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -expression(node.operand,values)
    if isinstance(node,ast.BinOp) and type(node.op) in OPS:return OPS[type(node.op)](expression(node.left,values),expression(node.right,values))
    if isinstance(node,ast.Compare) and len(node.ops)==1 and type(node.ops[0]) in OPS:return OPS[type(node.ops[0])](expression(node.left,values),expression(node.comparators[0],values))
    if isinstance(node,ast.IfExp):return expression(node.body if expression(node.test,values) else node.orelse,values)
    raise ValueError('Unsupported expression; no calls/imports/attributes/loops are executed')
def check(code,tests):
    try:
        tree=ast.parse(code);assert len(list(ast.walk(tree)))<100
        functions=[x for x in tree.body if isinstance(x,ast.FunctionDef)];assert len(functions)==1
        fn=functions[0];assert len(fn.body)==1 and isinstance(fn.body[0],ast.Return)
        assert [a.arg for a in fn.args.args]==['a','b']
        return True,all(expression(fn.body[0].value,{'a':a,'b':b})==expected for a,b,expected in tests)
    except Exception:return False,False
@torch.no_grad()
def evaluate_micro(model,cfg):
    model.eval();tok=Tokenizer.from_file(cfg['data']['tokenizer']);rows=[]
    for i,(name,reference,tests) in enumerate(TASKS):
        prompt=f'# Return the {name} of the two integers.\ndef prompt2_check_{i}_r7k3(a, b):\n    '
        x=torch.tensor([tok.encode(prompt).ids],device='cuda');generated=[]
        for _ in range(48):
            with amp(cfg):out=model(x)
            value=int(out['logits'][0,-1].argmax());generated.append(value)
            if value in (tok.token_to_id('<eos>'),tok.token_to_id('<doc>')):break
            x=torch.cat([x,torch.tensor([[value]],device='cuda')],1)
            if '\n\n' in tok.decode(generated,skip_special_tokens=True):break
        text=prompt+tok.decode(generated,skip_special_tokens=True).split('\n\n')[0];syntax,passed=check(text,tests)
        rows.append(dict(task_id=f'original_prompt2_{i}',prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                         reference_expression=reference,tests=tests,generated=text,syntax_and_supported_form=syntax,passed=passed))
    model.train();return dict(tasks=rows,pass_count=sum(r['passed'] for r in rows),tasks_count=len(rows),
                              scope='Six original deterministic arithmetic Python tasks, greedy48-token cap. Restricted arithmetic AST interpreter evaluates numeric test cases; model-generated Python is never exec/shell code. No benchmark-leadership or useful coding-agent claim.')
