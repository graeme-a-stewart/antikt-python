'''Benchmarking class to record standard results'''

import json
import platform
import sys

from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import run

@dataclass
class Benchmark:
    git_branch: str = run("git branch --show-current", capture_output=True, text=True).stdout.strip()
    git_hash: str = run("git rev-list --max-count=1 HEAD", capture_output=True, text=True).stdout.strip()
    timestamp: datetime = datetime.today()
    testname: str = Path(sys.argv[0]).name
    args: list = field(default_factory=list),
    nevents: int = 100
    runtimes: list = field(default_factory=list)
    version: str =  platform.python_version()
    platform: list = field(default_factory=dict)
    
    def __post_init__(self):
        self.args = sys.argv[1:]
        self.platform = platform.uname()._asdict()

    def to_json(self):
        '''Return a JSON version of our data'''
        me = {
            'git_branch': self.git_branch,
            'git_hash': self.git_hash,
            'timestamp': str(self.timestamp),
            'testname': self.testname,
            'args': self.args,
            'nevents': self.nevents,
            'runtimes': self.runtimes,
            'trials': len(self.runtimes),
            'version': self.version,
            'platform': self.platform,
        }
        return json.dumps(me, indent=2)
    