#!/bin/bash
for i in {0..19..1}
do
   f="../animal_game/models/21_08_20/noised_vectors/wiki_${i}*"
   g="../animal_game/models/21_08_20/noised_distance_matrices/wiki_${i}*"
   echo "Adding ${i}"
   git add $f
   git add $g
   git commit -a -m "add $i"
   git push
done

for i in {0..19..1}
do
   f="../animal_game/logs/21_08_20/individual/wiki_${i}*"
   g="../animal_game/logs/21_08_20/pairs/wiki_${i}*"
   echo "Adding ${i} log"
   git add $f
   git add $g
   git commit -a -m "add $i log"
   git push
done
