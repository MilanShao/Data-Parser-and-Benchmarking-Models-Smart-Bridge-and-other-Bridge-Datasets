## P-Bim Preprocessor
This tool allows to work with the P-Bim dataset. You can parse and plot data, transfer it into a binary intermediate representation and generate datasets from it.
### Usage
#### Parsing data
To parse data from a single group of measurements, run the following command:
```bash
    python3 bin/main.py parse <PATH_TO_DATA_DIR> <PATH_TO_OUTPUT_DIR> <JOB_NAME>
```
#### Plotting data
To plot data after parsing, run the following command:
```bash
    python3 bin/main.py plot <PATH_TO_PARSED_DATA_DIR> <PATH_TO_OUTPUT_DIR> <JOB_NAME> [--file <PARSED_DATA_FILE_NAME>]
```
If the `--file` option is not specified, all the files in the parsed data directory are plotted.
#### Processing data
Processing data into the intermediate format is possible via:
```bash
    python3 bin/main.py process <ZIP_FILE_PATH> <OUTPUT_PATH> [--workers <NUMBER_OF_WORKER_PROCESSES>] [--reset] [--tmp <TMP_DIRECTORY>]
```
The `--workers` option can be used to control the parallelism of the processing. It defaults to 8 if not specified. Beware that this process is relatively memory-hungry.<br/>
If the `--reset` flag is given, the directory given by `<OUTPUT_PATH>` is cleared before processing.<br/>
The `--tmp` option can be used to specify a temporary directory different from `/tmp/out`. This directory is used to unpack parts of the ZIP file into.<br/>
#### Assembling datasets
After processing, datasets can be generated from the intermediate representation via:
```bash
   python3 bin/main.py assemble <BASE_PATH> <OUTPUT_PATH> <START_TIME> <END_TIME> <RESOLUTION> [--strategy <SAMPLING_STRATEGY>] [--output-format <FORMAT>]
```
The directory specified by `<BASE_PATH>` should be the same as the one used as the `<OUTPUT_PATH>` in the `process` command. <br/>
Using the `--strategy` option you can control how data is sampled for the time series. Available options are `mean` and `interpolate`. The default strategy is `mean`.<br/>
Using the `--output-format` option you can alter the format of the final dataset. Available options are `csv` and `binary`. The default is `csv`.<br/>
The binary file has the following format:
```
    <file> ::= [<entry>]+
    <entry> ::= <time:i>[<channel:f>]+
```
The data is saved in little-endian and `:i` denotes 4 byte integers while `:f` denotes 4 byte float.
