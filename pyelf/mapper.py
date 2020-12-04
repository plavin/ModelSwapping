#-------------------------------------------------------------------------------
# elftools example: dwarf_decode_address.py
#
# Decode an address in an ELF file to find out which function it belongs to
# and from which filename/line it comes in the original source file.
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import print_function
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


def process_file(exefile, tracefile):
    print('Processing exe: {} and trace: {}'.format(exefile, tracefile))
    with open(exefile, 'rb') as f:
        elffile = ELFFile(f)

        if not elffile.has_dwarf_info():
            print('  file has no DWARF info')
            return

        # get_dwarf_info returns a DWARFInfo context object, which is the
        # starting point for all DWARF-based processing in pyelftools.
        dwarfinfo = elffile.get_dwarf_info()

        trace = sim.traceToInts(tracefile, None).IP
       
        counts = {}
        skip1 = 0
        skip2 = 0

        cache = {}
        unq = {}
        unqlist = {}

        mm = min(trace)
        print('Shifting all addresses by {}'.format(-mm))
        trace = np.array([a - mm for a in trace])

        for ip in trace:
            if ip in cache:
                counts[cache[ip]] += 1
                continue

            funcname = decode_funcname(dwarfinfo, ip+mm)
            file, line = decode_file_line(dwarfinfo, ip+mm)

            if funcname is None or file is None:
                funcstr = 'unknown'
            else:
                funcstr = '{}:{}'.format(bytes2str(file), bytes2str(funcname))

            cache[ip] = funcstr
            if funcstr in counts:
                counts[funcstr] += 1
                unq[funcstr] += 1
                unqlist[funcstr] = unqlist[funcstr] + [ip]
            else:
                counts[funcstr] = 1
                unq[funcstr] = 1
                unqlist[funcstr] = [ip]

        print('\nDynamic Counts: {} (total: {})'.format(counts, len(trace)))
        print('Static Counts: {} (total: {})'.format(unq, len(np.unique(trace))))
        range = {}
        strange = {} 
        for key in unqlist:
            if key !='unknown':
                vals = sorted(unqlist[key])
                range[key] = [vals[0], vals[-1]]
                strange[key] = []
                #print('\n{}: {}'.format(key, sorted(unqlist[key])))

        print()
        print(range)

        print()
        print('Checking for elements of unknown that are included in other ranges')

        for ip in unqlist['unknown']:
            for r in range:
                if ip >= range[r][0] and ip <= range[r][1]:
                    strange[r] = strange[r] + [ip]
                    break

        print(strange)

        print()
        print('Smallest 10 unknown addresses')
        print(sorted(unqlist['unknown'])[0:10])

def decode_funcname(dwarfinfo, address):
    # Go over all DIEs in the DWARF information, looking for a subprogram
    # entry with an address range that includes the given address. Note that
    # this simplifies things by disregarding subprograms that may have
    # split address ranges.
    for CU in dwarfinfo.iter_CUs():
        for DIE in CU.iter_DIEs():
            try:
                if DIE.tag == 'DW_TAG_subprogram':
                    lowpc = DIE.attributes['DW_AT_low_pc'].value

                    # DWARF v4 in section 2.17 describes how to interpret the
                    # DW_AT_high_pc attribute based on the class of its form.
                    # For class 'address' it's taken as an absolute address
                    # (similarly to DW_AT_low_pc); for class 'constant', it's
                    # an offset from DW_AT_low_pc.
                    highpc_attr = DIE.attributes['DW_AT_high_pc']
                    highpc_attr_class = describe_form_class(highpc_attr.form)
                    if highpc_attr_class == 'address':
                        highpc = highpc_attr.value
                    elif highpc_attr_class == 'constant':
                        highpc = lowpc + highpc_attr.value
                    else:
                        print('Error: invalid DW_AT_high_pc class:',
                              highpc_attr_class)
                        continue

                    if lowpc <= address <= highpc:
                        return DIE.attributes['DW_AT_name'].value
            except KeyError:
                continue
    return None


def decode_file_line(dwarfinfo, address):
    # Go over all the line programs in the DWARF information, looking for
    # one that describes the given address.
    for CU in dwarfinfo.iter_CUs():
        # First, look at line programs to find the file/line for the address
        lineprog = dwarfinfo.line_program_for_CU(CU)
        prevstate = None
        for entry in lineprog.get_entries():
            # We're interested in those entries where a new state is assigned
            if entry.state is None:
                continue
            if entry.state.end_sequence:
                # if the line number sequence ends, clear prevstate.
                prevstate = None
                continue
            # Looking for a range of addresses in two consecutive states that
            # contain the required address.
            if prevstate and prevstate.address <= address < entry.state.address:
                filename = lineprog['file_entry'][prevstate.file - 1].name
                line = prevstate.line
                return filename, line
            prevstate = entry.state
    return None, None


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Expected usage: {0} <exe> <trace>'.format(sys.argv[0]))
        sys.exit(1)
    process_file(sys.argv[1], sys.argv[2])
