train="Problem 2/training.txt"
test="Problem 2/testing.txt"
result="Problem2Results.txt"

D=$(cat "$train" | wc -l)
echo $D

rm $result
touch $result

for i in `seq 1 2 10`;
do
    echo $i $D "$train" "$test" | tee -a $result 
    python bagit.py $i $D "$train" "$test" | tee -a $result 
done
