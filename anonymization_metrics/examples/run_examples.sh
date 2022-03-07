
inputdir="scores/*"
outputdir="output"
mkdir "$outputdir"
fstat="$outputdir/stats.csv"

echo "i,matedMean,nonMatedMean,matedStd,nonMatedStd,linkability,cllr,cmin,eer" > $fstat


pyth="../compute_metrics.py"


parSteps=1 # parSteps is the # of parallel process
i=0 # process count

for s in $inputdir
do
    i=$(($i+1))
(
    tag=$(basename "$s")

    figureL="$outputdir/figure_linkability_${tag}"
    figureC="$outputdir/figure_cllr_${tag}"

    sOpt="$outputdir/opt_${tag}"
    if [ -f "$s" ]; then
        python3 $pyth -s ${s} -dl -ol ${figureL} -dc -oc ${figureC} -wopt -oopt ${sOpt}  --tag "${tag}" >> $fstat
        echo " $s done"
    else
        echo "Error file $s not found!"
        exit 1
    fi
  )&
  counter=$(($i%$parSteps))
  if [ $counter -eq 0 ]; then
    wait
  fi

done
wait

echo "All done, results in $(realpath $outputdir)"


