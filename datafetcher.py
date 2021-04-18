# Import relevant packages
import os, wget, sys
import pandas as pd

class PROCESS:
    def __init__(self, path):
        self.path = path

    def getData(self):
        metadata_path = self.path
        datapath = self.path+'/data'

        # Create data directory if not present
        if os.path.isdir(datapath):
            pass
        else:
            os.mkdir(datapath) 

        # Import metadata file
        metadata = pd.read_csv(metadata_path+'/metadata.tsv', sep='\t')

        for row in range(len(metadata.index)):
            url = metadata.loc[row]['File download URL']
            acc = metadata.loc[row]['File accession']
            if acc+'.bed.gz' in os.listdir(datapath):
                filename = wget.download(url, out=datapath)
            else:
                print('%s already downloaded.'%(acc))

        #print(metadata_path, datapath)

if __name__ == '__main__':
    path = sys.argv[1]
    obj = PROCESS(path)
    obj.getData()


    