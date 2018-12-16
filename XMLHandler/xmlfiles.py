"""This module provides a facade for accessing SemEval Task 3, Subtask B datasets."""

from zipfile import ZipFile

import wget

URLS = {
    "semeval2016-task3-cqa-ql-traindev-v3.2.zip": "http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-cqa-ql-traindev-v3.2.zip",
    "semeval2016_task3_test.zip": "http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016_task3_test.zip",
    "semeval2017_task3_test.zip": "http://alt.qcri.org/semeval2017/task3/data/uploads/semeval2017_task3_test.zip"}

XMLFNAMES = {
    2016 : {
        "train": [
            ("semeval2016-task3-cqa-ql-traindev-v3.2.zip", "v3.2/train/SemEval2016-Task3-CQA-QL-train-part1.xml"),
            ("semeval2016-task3-cqa-ql-traindev-v3.2.zip", "v3.2/train/SemEval2016-Task3-CQA-QL-train-part2.xml")],
        "dev": [("semeval2016-task3-cqa-ql-traindev-v3.2.zip", "v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml")],
        "test": [("semeval2016_task3_test.zip", "SemEval2016_task3_test/English/SemEval2016-Task3-CQA-QL-test.xml")]},
    2017 : {
        "test": [("semeval2017_task3_test.zip", "SemEval2017_task3_test/English/SemEval2017-task3-English-test.xml")]}}

class XMLFiles():
    def __init__(self, *path):
        obj = XMLFNAMES
        for segment in path:
            obj = obj[segment]
        self.zipfnames, self.xmlfnames = zip(*obj)
        self.zipfiles = []
        self.xmlfiles = []

    def __enter__(self):
        for zipfname, xmlfname in zip(self.zipfnames, self.xmlfnames):
            try:
                zipfile = ZipFile(zipfname)
            except IOError:
                assert wget.download(URLS[zipfname]) == zipfname
                zipfile = ZipFile(zipfname)
                self.zipfiles.append(zipfile)
            xmlfile = zipfile.open(xmlfname, 'r')
            self.xmlfiles.append(xmlfile)
        return self.xmlfiles

    def __exit__(self, type, value, traceback):
        for xmlfile in self.xmlfiles:
            xmlfile.close()
        for zipfile in self.zipfiles:
            zipfile.close()
