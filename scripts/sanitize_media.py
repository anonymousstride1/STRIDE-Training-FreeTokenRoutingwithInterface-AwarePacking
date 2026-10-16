"""Rewrite PDFs non-incrementally and remove image ancillary metadata."""
from pathlib import Path
import argparse
from pypdf import PdfReader, PdfWriter
from PIL import Image
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    for path in args.root.rglob('*'):
        if '.git' in path.parts or not path.is_file():continue
        if path.suffix.lower()=='.pdf':
            reader=PdfReader(path);writer=PdfWriter();writer.clone_document_from_reader(reader)
            writer.metadata=None
            writer._root_object.pop('/Metadata',None)
            for page in writer.pages:page.pop('/Metadata',None)
            tmp=path.with_suffix('.sanitized.pdf')
            with tmp.open('wb') as handle:writer.write(handle)
            tmp.replace(path)
            check=PdfReader(path);assert not check.metadata and '/Metadata' not in check.trailer['/Root']
            assert len(check.pages)==len(reader.pages)
        elif path.suffix.lower() in {'.png','.jpg','.jpeg'}:
            with Image.open(path) as im:
                fresh=Image.frombytes(im.mode,im.size,im.tobytes())
                tmp=path.with_name(path.stem+'.sanitized'+path.suffix)
                fresh.save(tmp)
            tmp.replace(path)
if __name__=='__main__':main()
