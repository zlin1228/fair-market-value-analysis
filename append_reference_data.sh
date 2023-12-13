CBONDS_DIR=$1
DATA_DIR=$2
head -2 ${CBONDS_DIR}/emissions.csv > ${DATA_DIR}/reference.csv
tail -n +3 -q ${CBONDS_DIR}/emissions*.csv >> ${DATA_DIR}/reference.csv
