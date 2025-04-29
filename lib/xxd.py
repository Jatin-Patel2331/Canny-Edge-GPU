#!/usr/bin/python3
#
# Copyright (c) 2023 Steffen Kieß
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Simple implementation of 'xxd -i -n'

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')
parser.add_argument('-i', action='store_true', required=True)
parser.add_argument('-n', required=True)
args = parser.parse_args()

# varname = args.n
varname = ''
for c in args.n:
    if c.isalnum():
        varname += c
    else:
        varname += '_'

with open(args.input_file, 'rb') as input:
    with open(args.output_file, 'w', encoding='utf-8') as output:
        output.write('unsigned char ' + varname + '[] = {\n')
        count = 0
        length = 0
        byte = input.read(1)
        while byte != b'':
            if count == 12:
                output.write(',\n  ')
                count = 0
            elif count == 0:
                output.write('  ')
            else:
                output.write(', ')
            print('0x{:02x}'.format(ord(byte)), file=output, end='')
            count += 1
            length += 1
            byte = input.read(1)
        if count != 0:
            output.write('\n')
        output.write('};\n')
        output.write('unsigned int ' + varname + '_len = {};\n'.format(length))
