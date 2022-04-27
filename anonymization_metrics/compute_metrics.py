from performance import cllr, min_cllr, ape_plot, linkability, draw_scores, writeScores, readScoresSingleFile
import argparse
import pandas as pd
import numpy as np




parser = argparse.ArgumentParser(description='Computing anonymizaiton metrics from pair score file, it outputs: "<tag>,matedMean,nonMatedMean,matedStd,nonMatedStd,linkaiblity,cllr,cmin,eer"')
parser.add_argument('-s', dest='score_file', type=str, nargs=1, required=True, help='path to score file')
parser.add_argument('-dc', dest='draw_ape_plot', action='store_true', help='flag: draw the APE-plot')
parser.add_argument('-dl', dest='draw_link_plot', action='store_true', help='flag: draw the Linkability Plot')
parser.add_argument('-oc', dest='output_file_cllr', type=str, nargs=1, required=False,   help='output path of the png and pdf file (default is ape_<score_file> ')
parser.add_argument('-ol', dest='output_file_link', type=str, nargs=1, required=False,   help='output path of the png and pdf file (default is link_<score_file> ')
parser.add_argument('--omega', dest='omega', type=float, nargs=1, required=False, default=1,   help='prior ratio for linkability metric (default is 1)')
parser.add_argument('--tag', dest='tag', type=str, nargs=1, required=False, default="results", help='Tag before the values in the order of linkaiblity,cllr,cmin,eer')
parser.add_argument('-wopt', dest='write_opt', action='store_true', help='flag: Write the calibrated scores')
parser.add_argument('-oopt', dest='output_file_opt', type=str, nargs=1, required=False,   help='output path of the calibrated scores (default is opt_<score_file> ')


args = parser.parse_args()

matedScores, nonMatedScores = readScoresSingleFile(args.score_file[0])

cllr = cllr(matedScores, nonMatedScores)
cmin, eer, matedScores_opt, nonMatedScores_opt = min_cllr(matedScores, nonMatedScores,compute_eer=True, return_opt=True)
Dsys, D, bin_centers, bin_edges = linkability(matedScores, nonMatedScores, args.omega)




if args.draw_ape_plot:
  output_file= "ape_"+args.score_file[0]
  if args.output_file_cllr is not None:
    output_file = args.output_file_cllr[0]
  ape_plot(matedScores, nonMatedScores, matedScores_opt, nonMatedScores_opt, cllr, cmin, eer, output_file)


if args.write_opt:
  output_file= "opt_"+args.score_file[0]
  if args.output_file_opt is not None:
    output_file = args.output_file_opt[0]
  writeScores(matedScores_opt, nonMatedScores_opt, output_file)


if args.draw_link_plot:
  output_file= "linkability_"+args.score_file[0]
  if args.output_file_link is not None:
    output_file = args.output_file_link[0]
  draw_scores(matedScores, nonMatedScores, Dsys, D, bin_centers, bin_edges, output_file)


matedMean = np.mean(matedScores)
matedStd = np.std(matedScores)

nonMatedMean = np.mean(nonMatedScores)
nonMatedStd = np.std(nonMatedScores)


print("{0},{1},{2},{3},{4},{5},{6},{7},{8}".format(args.tag[0],matedMean,nonMatedMean,matedStd,nonMatedStd,Dsys,cllr,cmin,eer))

