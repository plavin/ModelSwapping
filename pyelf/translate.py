import sys

# If pyelftools is not installed, the example can also run from the root or
# examples/ dir of the source distribution.
sys.path[0:0] = ['.', '..']
sys.path.insert(1, '../')
import sveCacheSim as sim
import numpy as np

from elftools.common.py3compat import maxint, bytes2str
from elftools.dwarf.descriptions import describe_form_class
from elftools.elf.elffile import ELFFile
from typing import List

def print_dict(d):
    for k in d:
        print('{} ->\n\t{}'.format(k,d[k]))

if len(sys.argv) < 3:
    print('Usage:\n  python3 {} <exefile> <tracefile>'.format(sys.argv[0]))
    exit()

offset_map = {}


class range:
    def __init__(self, name:str, start:np.uint64, end:np.uint64, ref:np.int64=-1):
        self.name = name
        self.start = start
        self.end = end
        self.ref = ref
        self.child = []

    def contains(self, new):
        return new.start >= self.start and new.end <= self.end
    def has(self, addr):
        return addr >= self.start and addr <= self.end
    def _tostring(self, level):
        spaces = '' if level == 0 else '{}↳ '.format(' '*(level)*2)
        res = '{}{} [0x{:x} - 0x{:x}]\n'.format(spaces, self.name, self.start, self.end)
        for c in self.child:
            res = res + c._tostring(level+1)
        return res
    def __str__(self):
        return self._tostring(0)

def insert(root:range, new:range):
    for c in root.child:
        if c.contains(new):
            insert(c, new)
            return
    root.child.append(new)

def fixup(root:range, map):
    if root.name is None:
        root.name = map[root.ref]
    for c in root.child:
        fixup(c, map)

def find(root:range, addr:np.uint64):
    name = root.name
    for c in root.child:
        if c.has(addr):
            name = find(c, addr)
    return name

root = range('unknown', 0,0xffffffffffffffff)

with open(sys.argv[1], 'rb') as exefile:
    elffile = ELFFile(exefile)
    if not elffile.has_dwarf_info():
        print('Error: {} has no dwarf info'.format(sys.argv[1]))
        exit()

    dwarfinfo = elffile.get_dwarf_info()
    for CU in dwarfinfo.iter_CUs():
        for DIE in CU.iter_DIEs():
            try:
                if DIE.tag == 'DW_TAG_inlined_subroutine':
                    offset = DIE.attributes['DW_AT_abstract_origin'].value
                    start  = DIE.attributes['DW_AT_low_pc'].value
                    end    = DIE.attributes['DW_AT_low_pc'].value + DIE.attributes['DW_AT_high_pc'].value

                    insert(root, range(None, start, end, offset))

                    print('  Inlined subroutine [[[off:{}]]]'.format(DIE.attributes['DW_AT_abstract_origin'].value), end='')
                    if 'DW_AT_low_pc' in DIE.attributes:
                        print(' [{} - {}]'.format(DIE.attributes['DW_AT_low_pc'].value, DIE.attributes['DW_AT_high_pc'].value))
                    else:
                        print(' COULDNT FIND RANGE')
                elif DIE.tag == 'DW_TAG_subprogram':

                    name = DIE.attributes['DW_AT_name'].value.decode('UTF-8')
                    print('Subroutine [[[{}]]]'.format(name), end='')
                    offset_map[DIE.offset] = name

                    start  = DIE.attributes['DW_AT_low_pc'].value
                    end    = DIE.attributes['DW_AT_low_pc'].value + DIE.attributes['DW_AT_high_pc'].value


                    insert(root, range(name, start, end, ))
                    #print_dict(DIE.attributes)
                    if 'DW_AT_low_pc' in DIE.attributes:
                        print(' [{} - {}]'.format(DIE.attributes['DW_AT_low_pc'].value, DIE.attributes['DW_AT_high_pc'].value))
                    else:
                        print(' COULDNT FIND RANGE of DIE offset {}'.format(DIE.offset))
            except KeyError:
               continue
        #print(CU._dielist)
        #print(CU.header)
        cu_file = dwarfinfo.line_program_for_CU(CU)['file_entry'][0].name.decode('UTF-8')
        print(cu_file)
        print(offset_map)

fixup(root, offset_map)
print(root)


trace = sim.traceToInts(sys.argv[2], None).IP

counts = {}
counts['unknown'] = 0
for k in offset_map:
    counts[offset_map[k]] = 0
for ip in trace:
    counts[find(root, ip)] += 1

print(counts)

