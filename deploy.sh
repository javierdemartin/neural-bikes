# Remove unused model files generated during testing
rm -rf model/*/*/

rm -rf __pycache__/

rm -rf plots/

rm -rf data/*/filled/

rm -rf data/*/supervised/

rm -rf data/*/tomorrow/

rm -rf data/*/scaled/

rm -rf data/*/encoded_data/

rm -rf data/*/cluster/*/

rm -rf data/*/cluster/distortions.csv

rm *.out

find . -name '.DS_Store' -type f -delete

scp -rp model javierdemartin@192.168.86.99:/Users/javierdemartin/Documents/neural-bikes/model