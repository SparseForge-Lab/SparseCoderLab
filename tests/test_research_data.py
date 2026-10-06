import copy
from tools.research_corpus import candidate,split

SOURCE={'dataset':'codeparrot/github-code','revision':'pinned','remote_file':'data/test.parquet'}
def row():return {'content':'def calculate_value(x):\n    y = x + 1\n    return y\n'+'# explanatory comment\n'*6,'repo_name':'Owner/Project','path':'src/main.py','license':'mit'}
def test_real_raw_schema_and_repository_split():
    a,error=candidate(row(),SOURCE,1);assert error is None and a['language']=='Python'
    b=row();b['path']='lib/other.py';b['content']+='\n# different file\n'
    second,_=candidate(b,SOURCE,2)
    assert a['split']==second['split']==split('github:owner/project')
    assert a['revision']=='pinned' and a['row_index']==1
def test_unknown_and_vendored_licenses_rejected():
    for change,reason in [({'license':'unknown'},'license'),({'path':'node_modules/lib/main.py'},'vendored_or_generated_path')]:
        value=row();value.update(change);assert candidate(value,SOURCE,1)==(None,reason)
def test_normalized_duplicate_key_without_changing_source():
    a,_=candidate(row(),SOURCE,1);b=row();b['content']=b['content'].replace('\n','\r\n')+'\n'
    other,_=candidate(b,SOURCE,2)
    assert a['normalized_sha256']==other['normalized_sha256']
    assert a['source_content_sha256']!=other['source_content_sha256']
def test_permissive_repository_conflicting_header_rejected():
    value=row();value['content']='/* GNU GENERAL PUBLIC LICENSE */\n'+value['content']
    assert candidate(value,SOURCE,1)==(None,'conflicting_license_header')
