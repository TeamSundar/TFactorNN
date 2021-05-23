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
                print('\n%s already downloaded.'%(acc))
            else:
                print('\nDownloading %s'%acc)
                filename = wget.download(url, out=datapath)
        return None

    def importMetadata(self):
        # Prepare metadata
        metadata = pd.read_csv(self.path+'/metadata.tsv', sep='\t')
        metadata_ = metadata[['File accession', 'Biosample term name', 'Biosample type', 'File assembly', 'Experiment target']]
        metadata_ = metadata_[metadata_['Biosample type']=='cell line']
        return metadata_

    def filterDuplicate(self):
        metadata = self.importMetadata()
        # Filte duplicate tf-celline pairs
        cellline_dict = collections.Counter(metadata_['Biosample term name'])
        master_record = pd.DataFrame()
        for line in tqdm(cellline_dict):
            df = metadata_[metadata_['Biosample term name']==line]
            tfs = collections.Counter(df['Experiment target'])
            for tf in tfs:
                df_ = df[df['Experiment target']==tf]
                master_record = master_record.append(df_[df_['File accession']==random.choice(df_['File accession'].values)], ignore_index=True)
        print(master_record.shape)
        return master_record

    def zip2txt():
        pass
        
    def dataSummary(self):
        master_record = self.filterDuplicate()
        for acc in tqdm(master_record['File accession'].values):
            try:
                data = pd.read_csv('/data/'+acc+'.bed', sep='\t', header=None)
                list_ = abs(data[1].values-data[2].values)
                means.append(np.mean(list_))
            except:
                print(acc)
        plt.hist(means, bins=30)
        print(stats.describe(means))
        #print(metadata_path, datapath)
        return None