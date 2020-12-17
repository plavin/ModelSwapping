import sys
sys.path.insert(1, '../')
import sveCacheSim as sim
import numpy as np
from elftools.elf.elffile import ELFFile #conda install pyelftools

# HOW TO USE:
# An example is at the bottom of this file, in __main__.
# Basically, you just need to create a DWARFMap object by giving it
# the executable with dwarf info relating to your trace (e.g. compile with -g).
# Then, you can pass a trace into the classify function and you will get a
# dictonary back. This dict has keys that are function names and values that
# is the number of instructions from the trace corresponding to each function.

# A range represents a range of 64 bit addresses. Each one will be named for a
# function. Each can optionally hold a reference, which is the offset of a DIE
# that contains the function name associated with this range. This allows us
# to fill in the name later, if we don't know it right away (as is the case
# with inlined functions)
class range:
    def __init__(self, name:str, start:np.uint64, end:np.uint64, ref:np.int64=-1):
        self.name = name
        self.start = start
        self.end = end
        self.ref = ref
        self.child = []

    # Returns true if the range passed as an argument is completely contained within
    # this range (inclusive) and false otherwise
    def contains(self, new:range):
        return new.start >= self.start and new.end <= self.end


    # Returns true if the address passed as an argument is within this range (inclusive)
    # and false otherwise
    def has(self, addr:np.uint64):
        return addr >= self.start and addr <= self.end

    # Use the map to figure out the names of inlined functions
    def map_inlined(self, map):
        if self.name is None:
            if self.ref in map:
                self.name = map[self.ref]
            else:
                self.name = 'unkown2' # TODO: figure this out (try running meabo)
        for c in self.child:
            c.map_inlined(map)

    # Add a new range. Insert it at the narrowest possible spot, i.e. if it
    # is contained within another range, it must be place within it.
    def insert(self, new:range):
        for c in self.child:
            if c.contains(new):
                c.insert(new)
                return
        self.child.append(new)

    # Find the lowest/narrowest range that the address is contained in
    def find(self, addr:np.uint64):
        name = self.name
        for c in self.child:
            if c.has(addr):
                name = c.find(addr)
        return name

    # Helper function for __str__ that lets us do indentation
    def _tostring(self, level):
        spaces = '' if level == 0 else '{}↳ '.format(' '*(level)*2)
        res = '{}{} [0x{:x} - 0x{:x}]\n'.format(spaces, self.name, self.start, self.end)
        for c in self.child:
            res = res + c._tostring(level+1)
        return res

    def __str__(self):
        return self._tostring(0)

class DWARFMap:
    def __init__(self, file):
        # The root node of our tree contains all possible addresses
        # and is named unknown, as this is what is returned when
        # we don't have DWARF info for an address
        self.root = range('unknown', 0,0xffffffffffffffff)

        # Stores the offsets of all DIEs that define functions, so that we can later
        # discern the names of inlined functions, which do not store the function
        # name at their inlined location
        self.offset_map = {}

        # Use pyelftools to get an object containing all DWARF info

        with open(file, 'rb') as exefile:
            elffile = ELFFile(exefile)
            if not elffile.has_dwarf_info():
                print('Error: {} has no dwarf info'.format(file))
                exit()
            dwarfinfo = elffile.get_dwarf_info()

            # Iterate over every compute unit (roughtly every input .c/.cpp file)
            for CU in dwarfinfo.iter_CUs():

                # Iterate over every Debugging Information Entry and search for ones
                # that represent subroutines
                for DIE in CU.iter_DIEs():
                    try:
                        if DIE.tag == 'DW_TAG_inlined_subroutine':
                            offset = DIE.attributes['DW_AT_abstract_origin'].value
                            start  = DIE.attributes['DW_AT_low_pc'].value
                            end    = DIE.attributes['DW_AT_low_pc'].value + DIE.attributes['DW_AT_high_pc'].value

                            # Insert a node with this range in the tree. It currently has no name (None) as it was an inlined
                            # function. We will get the name later when we find the DIE with the corresponding
                            # offset, which this DIE lists as DW_AT_abstract_origin
                            self.root.insert(range(None, start, end, offset))

                        elif DIE.tag == 'DW_TAG_subprogram':

                            # Go ahead and store the name and offset of this DIE, as it represents a function. We will need
                            # this map later when we want to get the names of inlined functions
                            name = DIE.attributes['DW_AT_name'].value.decode('UTF-8')
                            self.offset_map[DIE.offset] = name

                            # If the DIE has a low_pc, it should also have a high_pc. If it has neither, we
                            # can't determine a range so we will continue
                            if 'DW_AT_low_pc' not in DIE.attributes:
                                continue
                            start  = DIE.attributes['DW_AT_low_pc'].value
                            end    = DIE.attributes['DW_AT_low_pc'].value + DIE.attributes['DW_AT_high_pc'].value

                            # Insert a node with this range in the tree. If we have made it this far it means
                            # we have a start and an end address, meaning this is a full function, and not
                            # just a prototype.
                            self.root.insert(range(name, start, end))
                    except KeyError:
                        continue #TODO figure out what this is necesseary

            self.root.map_inlined(self.offset_map)

    def classify(self, ips, counts={}):
        if np.isscalar(ips):
            return self.root.find(ips)
        # Initialize counts if it is empty
        # Subsequent calls can pass in counts and
        # we will just add to that one.
        if not counts:
            counts['unknown'] = 0
            for k in self.offset_map:
                counts[self.offset_map[k]] = 0
        for ip in ips:
            counts[self.root.find(ip)] += 1
        return counts


if __name__ == '__main__':
    if len(sys.argv) < 2 or (len(sys.argv) == 2 and sys.argv[1] == '-h'):
        print('Usage:\n  python3 {} <exefile> [tracefile]'.format(sys.argv[0]))
        exit()

    # Parse DWARF
    DM = DWARFMap(sys.argv[1])
    # This prints out the hierarcical ranges found in the DWARF
    print(DM.root, end='')

    if len(sys.argv) > 2:
        # Use sveCacheSim to load the trace
        trace = sim.traceToInts(sys.argv[2], None).IP
        # This will attribute each IP in the trace to a function
        counts = DM.classify(trace)
        print(counts)

