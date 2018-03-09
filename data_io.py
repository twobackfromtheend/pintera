import numpy as np
import os
# import matplotlib.pyplot as plt
import struct
import sys


def get_delta(f, v=True):
    offsets = [700, 590]
    # open data file in HxD, and find the offset of the first 2 pairs after the long 00s. add ints here if needed.
    # I found 700 to be the offset for saved files from the Lab's computer
    # 590 was after the file was converted to a newer file type on my laptop by LabView 2017
    file = open(f, mode='rb')
    found_deltas = []

    # find deltas in file at offsets above
    for offset in offsets:
        file.seek(offset)
        delta = struct.unpack('>d', file.read(8))[0]
        found_deltas.append(delta)

    # print(found_deltas)

    # filter deltas for integer and non-zero
    found_deltas = [int(d) for d in found_deltas if d == int(d) and d != 0 and d < 1e3]

    if len(found_deltas) > 1:
        print('Error: Found 2 deltas (%s) in file %s.' % (found_deltas, os.path.basename(f)))
    if v:
        print("Found delta %s for %s." % (found_deltas[0], os.path.basename(f)))
    return found_deltas[0]


def get_y(f):
    y_offset = 89 * 8
    y_offset_new = 2 + 76 * 8
    # y_offset_new is for files converted by LabView 2017.
    # (you'll also need to cut off the last 3 instead of the last 4 below)
    file = open(f, 'rb')
    file.seek(y_offset)
    y = np.fromfile(file, dtype='>d')
    y = y[:-4]  # change to -3 for new format files

    # try using new format stuff if max > 100  (hacky check)
    if max(y) > 100:
        file = open(f, 'rb')
        file.seek(y_offset_new)
        y = np.fromfile(file, dtype='>d')
        y = y[:-3]

    return y


def get_x_y_delta(f, v=True):
    _delta = get_delta(f, v)
    _y = get_y(f)

    # create x from deltas
    _x = np.arange(len(_y)) * _delta
    return _x, _y, _delta


def save_to_csv(f):
    _x, _y, _delta = get_x_y_delta(f)

    data = np.column_stack((_x, _y))
    new_file_name = f + '.csv'
    np.savetxt(new_file_name, data, fmt=('%i', '%.18e'), delimiter=',')
    print('Saved %s to %s.' % (os.path.basename(f), os.path.basename(new_file_name)))
    return new_file_name, _x, _y, _delta


def parse_dropped_input():
    file_paths = sys.argv[1:]
    input_files_count = len(file_paths)
    if input_files_count == 0:
        print("No input received.")
        print("Drag and drop file(s) (that are not .csvs) to parse them to csvs.")
        input("Press enter to continue...")

    else:
        log = ''

        non_csvs = [file_path for file_path in file_paths if not file_path.endswith(".csv")]
        non_csvs_count = len(non_csvs)

        _str = "Parsing %s/%s files. (Only parsing non .csv files)" % (non_csvs_count, input_files_count)
        print(_str)
        log += '\n' + _str
        log += '\nFull File list:' + str(file_paths)

        successes = 0
        failures = 0
        for file_path in non_csvs:
            _str = "Trying to parse %s" % os.path.basename(file_path)
            print(_str)
            log += '\n' + _str

            try:
                _, _, _, _delta = save_to_csv(file_path)

                _str = "Successfully parsed %s (delta: %s)" % (os.path.basename(file_path), _delta)
                print(_str)
                log += '\n' + _str

                successes += 1
            except Exception as e:
                failures += 1
                _str = "Failed to parse %s: %s" % (os.path.basename(file_path), e)
                print(_str)
                log += '\n' + _str

        _str = 'Successfully parsed %s files (failed %s).' % (successes, failures)
        print(_str)
        log += '\n\n' + _str

        with open('log.txt', 'w') as f:
            f.write(log)

        print("Log written to log.txt.")
        input("Press enter to continue...")


if __name__ == '__main__':
    parse_dropped_input()

    # The below is an example for plotting and showing multiple data files. It will save the last as a csv.
    # (uncomment the import matplotlib above)

    # data_dir = 'RSAHX'  # change to blank string ('') if data is not in its own directory
    # # file_names = ['10S2new']
    # # file_names = ['10S2old']
    # file_names = ['20S3old', '10S2old', '10S2new', 'W1S2', 'OPW5S1']
    #
    # for file_name in file_names:
    #     data_file = os.path.join(data_dir, file_name)
    #     x, y, delta = get_x_y_delta(data_file)
    #     plt.plot(x, y)
    #     plt.show()
    #
    # # save_to_csv(data_file)
