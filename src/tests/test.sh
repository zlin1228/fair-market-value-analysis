#!/bin/bash

# make sure the first argument is one of "unit", "integration", or "all"
if [[ $1 != 'unit' && $1 != "integration" && $1 != "all" ]]; then
    echo 1>&2 "invalid first argument: $1"
    INVALID_ARGS=true
fi

# if there is a second argument make sure it is "coverage" and set $COVERAGE=true
if [ $# -gt 1 ]; then
    if [ $2 == "coverage" ]; then
        COVERAGE=true
    else
        echo 1>&2 "invalid second argument: $2"
        INVALID_ARGS=true
    fi
fi

if [[ $INVALID_ARGS || $# -lt 1 ]]; then
    echo 1>&2 "usage: $ test.sh {unit|integration|all} [coverage]"
    echo 1>&2 "    example: run the unit tests with code coverage"
    echo 1>&2 "      $ test.sh unit coverage"
    echo 1>&2 "    example: run the integration tests without code coverage"
    echo 1>&2 "      $ test.sh integration"
    echo 1>&2 "    example: run all the tests with code coverage"
    echo 1>&2 "      $ test.sh all coverage"
    exit 2
fi

EXIT_CODE=0

# clear artifacts from previous test runs (ignore errors)
echo "removing previous artifacts"
coverage erase
rm output_*.txt
rm -r ./htmlcov

echo "running test(s) in parallel... (output piped to 'output_*' files)"

# run the unit tests if first argument is "unit" or "all"
if [[ $1 == "unit" || $1 == "all" ]]; then
    # run with coverage if second argument is "coverage" otherwise run with python
    if [[ $COVERAGE ]]; then
        coverage run --parallel-mode --source=../ --branch -m unittest &> output_unit_tests.txt & unit_tests=$!
    else
        python -m unittest &> output_unit_tests.txt & unit_tests=$!
    fi
    echo "unit tests running in child process $unit_tests"
fi
# run the integration tests if first argument is "integration" or "all"
if [[ $1 == "integration" || $1 == "all" ]]; then
    # run with coverage if second argument is "coverage" otherwise run with python
    if [[ $COVERAGE ]]; then
        coverage run --parallel-mode --source=../ --branch integration_test_update_model_data.py &> output_integration_test_update_model_data.txt & integration_test_update_model_data=$!
        coverage run --parallel-mode --source=../ --branch integration_test_train_evaluate.py &> output_integration_test_train_evaluate.txt & integration_test_train_evaluate=$!
        coverage run --parallel-mode --source=../ --branch integration_test_grouped_history.py &> output_integration_test_grouped_history.txt & integration_test_grouped_history=$!
        coverage run --parallel-mode --source=../ --branch integration_test_fmv_websocket_server.py &> output_integration_test_fmv_websocket_server.txt & integration_test_fmv_websocket_server=$!
        coverage run --parallel-mode --source=../ --branch integration_test_load_data.py &> output_integration_test_load_data.txt & integration_test_load_data=$!
    else
        python integration_test_update_model_data.py &> output_integration_test_update_model_data.txt & integration_test_update_model_data=$!
        python integration_test_train_evaluate.py &> output_integration_test_train_evaluate.txt & integration_test_train_evaluate=$!
        python integration_test_grouped_history.py &> output_integration_test_grouped_history.txt & integration_test_grouped_history=$!
        python integration_test_fmv_websocket_server.py &> output_integration_test_fmv_websocket_server.txt & integration_test_fmv_websocket_server=$!
        python integration_test_load_data.py &> output_integration_test_load_data.txt & integration_test_load_data=$!
    fi
    echo "integration tests running in child processes $integration_test_load_data, $integration_test_fmv_websocket_server, $integration_test_grouped_history, $integration_test_train_evaluate, $integration_test_update_model_data"
fi

report_result () {
    # EXIT_CODE will only remain 0 if all child processes exit with 0
    if [ $2 -ne 0 ]; then EXIT_CODE=$2; fi
    # report whether the test from the child process passed or failed
    if [ $2 == 0 ]; then result="pass"; else result="FAIL"; fi
    echo "$result: $1"
}

# try to keep these in order of how long they take to finish
# so we get results as they finish instead of getting all
# the results at the end
# wait on each child process and report the result
if [[ $1 == "unit" || $1 == "all" ]]; then
    wait $unit_tests
    report_result "unit tests" $?
fi
if [[ $1 == "integration" || $1 == "all" ]]; then
    wait $integration_test_load_data
    report_result "integration_test_load_data" $?
    wait $integration_test_fmv_websocket_server
    report_result "integration_test_fmv_websocket_server" $?
    wait $integration_test_grouped_history
    report_result "integration_test_grouped_history" $?
    wait $integration_test_train_evaluate
    report_result "integration_test_train_evaluate" $?
    wait $integration_test_update_model_data
    report_result "integration_test_update_model_data" $?
fi

if [[ $COVERAGE ]]; then
    # this combines the coverage data from all of the tests
    coverage combine
    # creates an html report of the coverage data available at src/tests/htmlcov/index.html
    coverage html
fi

if [[ $EXIT_CODE == 0 ]]; then echo "test results: pass!"; else echo "test results: FAIL"; fi
exit $EXIT_CODE
