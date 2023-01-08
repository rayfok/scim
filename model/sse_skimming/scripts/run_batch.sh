#!/bin/bash

### TEST SET ###
# NAACL 2022: 1 - 443 (443)
# ================================
# TOTAL: 443 papers

### TRAIN SET ###
# NAACL 2021: 1 - 477 (477)
# NAACL 2019: 1001 - 1424 (424)
# NAACL 2018: 1001 - 1205 (205)
# ACL 2022: 1 - 603 (603)
# ACL 2021: 1 - 571 (571)
# ACL 2020: 1 - 778 (778)
# ================================
# TOTAL: 3,058

START=$1
END=$2

for (( i=$START; i<=$END; i++ ))
do
    # NAACL22="https://aclanthology.org/2022.naacl-main.$i.pdf"
    # NAACL21="https://aclanthology.org/2021.naacl-main.$i.pdf"
    NAACL19="https://aclanthology.org/N19-$i.pdf"
    # NAACL18="https://aclanthology.org/N18-$i.pdf"
    # ACL22="https://aclanthology.org/2022.acl-long.$i.pdf"
    # ACL21="https://aclanthology.org/2021.acl-long.$i.pdf"
    # ACL20="https://aclanthology.org/2020.acl-main.$i.pdf"

    python -m sse_skimming \
            src=$NAACL19 \
            dst=./output
done
