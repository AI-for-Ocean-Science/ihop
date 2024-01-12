# s3

## Generate bucket

aws s3api create-bucket --bucket ihop --endpoint-url https://s3-west.nrp-nautilus.io

## Policy

aws s3api put-bucket-policy --bucket ihop --policy file://s3_ihop_policy.json --endpoint https://s3-west.nrp-nautilus.io

## Check users

aws s3api  get-bucket-acl --bucket ihop  --endpoint https://s3-west.nrp-nautilus.io