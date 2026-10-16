"""Check every released file, including decoded media, for identity and secrets.

Identity expressions are supplied outside the repository to avoid publishing them.
Success prints nothing and returns zero.
"""
from pathlib import Path
import argparse, re, sys, json
from pypdf import PdfReader
from PIL import Image
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--pattern',required=True);args=p.parse_args()
    identity=re.compile(args.pattern,re.I)
    secret=re.compile(r'gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    assignments=re.compile(r'''(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|wandb[_-]?api[_-]?key)\b['"]?\s*[:=]\s*['"][^'"\s]{8,}['"]''')
    local=re.compile(r'[A-Z]:[\\/](?:Users|STRIDE)[\\/]|/(?:home|Users)/[^/\s]+',re.I)
    failures=[]
    for path in args.root.rglob('*'):
        if path.is_symlink():
            failures.append((str(path.relative_to(args.root)),'symbolic link'))
            continue
        if '.git' in path.parts or not path.is_file():continue
        name=path.relative_to(args.root).as_posix()
        if identity.search(name):failures.append((name,'filename identity'))
        if path.stat().st_size>95*1024*1024:failures.append((name,'oversized'))
        payload=path.read_bytes().decode('utf-8',errors='ignore')
        if path.suffix=='.py' and path.name=='check_release.py':
            # Expressions describe scanners, not credential values.
            secret_payload='\n'.join(line for line in payload.splitlines() if 're.compile(' not in line)
        else:secret_payload=payload
        if identity.search(payload):failures.append((name,'content identity'))
        if secret.search(secret_payload):failures.append((name,'secret candidate'))
        if assignments.search(secret_payload):failures.append((name,'credential assignment'))
        if local.search(payload):failures.append((name,'local path'))
        if path.suffix=='.ipynb':
            nb=json.loads(payload)
            if nb.get('metadata') or any(c.get('outputs') or c.get('execution_count') is not None or c.get('metadata') for c in nb['cells']):failures.append((name,'notebook metadata/output'))
        if path.suffix.lower()=='.pdf':
            reader=PdfReader(path)
            decoded='\n'.join(page.extract_text() or '' for page in reader.pages)
            if identity.search(decoded):failures.append((name,'decoded PDF identity'))
            if reader.metadata or '/Metadata' in reader.trailer['/Root']:failures.append((name,'PDF metadata'))
            for page in reader.pages:
                if '/Metadata' in page:failures.append((name,'page metadata'))
            if reader.attachments:failures.append((name,'embedded attachments'))
            if reader.outline and identity.search(str(reader.outline)):failures.append((name,'outline identity'))
            # Decode indirect objects, including URI annotations and compressed streams.
            from pypdf.generic import IndirectObject, StreamObject
            for generation,objects in reader.xref.items():
                for number in objects:
                    if number==0:continue
                    obj=IndirectObject(number,generation,reader).get_object()
                    decoded_obj=str(obj)
                    if isinstance(obj,StreamObject):decoded_obj+=obj.get_data().decode('utf-8',errors='ignore')
                    if identity.search(decoded_obj) or local.search(decoded_obj):failures.append((name,'PDF object identity/path'))
                    if secret.search(decoded_obj):failures.append((name,'PDF object secret'))
        if path.suffix.lower() in {'.png','.jpg','.jpeg'}:
            with Image.open(path) as im:
                if im.getexif() or any(k in im.info for k in ['xmp','exif','comment','Description','Author']):failures.append((name,'image metadata'))
    for name,reason in failures:print(f'{name}: {reason}',file=sys.stderr)
    raise SystemExit(bool(failures))
if __name__=='__main__':main()
