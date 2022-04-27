from gen_pseudo_xvecs import train_models, save_pca_and_gmm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('anoni_pool')
    parser.add_argument('xvec_out_dir')
    parser.add_argument('pickle_file')

    args = parser.parse_args()
    models = train_models(args.anoni_pool, args.xvec_out_dir)
    save_pca_and_gmm(models, args.pickle_file)
