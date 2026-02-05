# THIS CODE MUST RUN ON ALIENWARE OR THE LOCAL PC (WITH SOME UN-COMMENT) BECAUSE OF THE ACCESS TO THE DATASET
from PIL import Image
import os
import shutil


def clear_tiles_directory(dataset_name, dataset_path, tile_measure):
    if os.path.exists(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}'):
        print('Previous tiles erasing...')
        shutil.rmtree(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}')
        print('DONE!\n\n\n')
    os.makedirs(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}')


def tiles_creation(dataset_name, dataset_path, tile_measure, maps_to_use):

    os.makedirs(f'{dataset_path}/{dataset_name}/tiles', exist_ok=True)

    for dataset_type in ['train', 'val']:

        print(f'Dataset type ------- {dataset_type}')

        gt = True
        #if not os.path.exists(f'/home/antoniocorvino/Projects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/gt'):
        #if not os.path.exists( f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/gt'):
        if not os.path.exists(f'/mnt/nas151/sar/Footprint/datasets/{dataset_name}/tiles/{dataset_type}/gt'):

            print(f'---------{dataset_type} dataset has not Ground Truth---------')
            gt = False

        os.makedirs(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}/{dataset_type}/', exist_ok=True)
        os.makedirs(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}/{dataset_type}/gt', exist_ok=True)

        full_maps = sorted(os.path.splitext(f)[0] for f in os.listdir(f'{dataset_path}/{dataset_name}/tiles/{dataset_type}/images') if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.TIF')))

        print(f'Maps used in {dataset_type}: {maps_to_use}/{len(full_maps)}..................')

        full_maps = full_maps[:maps_to_use]


        for name in full_maps:
            #image = Image.open(f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/images/{name}.tif')
            #image = Image.open(f'/home/antoniocorvino/Projects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/images/{name}.tif')
            image = Image.open(f'/mnt/nas151/sar/Footprint/datasets/{dataset_name}/tiles/{dataset_type}/images/{name}.TIF')
            if gt:
                #mask = Image.open(f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/gt/{name}.tif')
                #mask = Image.open(f'/home/antoniocorvino/Projects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/gt/{name}.tif')
                mask = Image.open(f'/mnt/nas151/sar/Footprint/datasets/{dataset_name}/tiles/{dataset_type}/gt/{name}.tif')

            count = 0

            height, width = image.size

            for i in range(0, height, tile_measure):
                for j in range(0, width, tile_measure):
                    count += 1
                    box = (j, i, j + tile_measure, i + tile_measure)

                    if gt:
                        mask_tile = mask.crop(box)

                        #if (mask_tile.size != (tile_measure, tile_measure) or
                        #    mask_tile.getextrema() == ((0, 0), (0, 0), (0, 0)) or
                        #    mask_tile.getextrema() == (0, 0)):                      # VERIFICARE COME MAI I DUE DATASET SALVANO IN MODO DIVERSO LE MASCHERE, UNO RGB E UNA BW

                        #    skipped += 1
                        #    continue
                    #print('Tiles skipped: {skipped}/{count} ({(100 * skipped / count):.1f}%)\n\n')
                    image_tile = image.crop(box)
                    image_tile.save(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}/{dataset_type}/images/{name}_{count}.tif')
                    if gt:
                        mask_tile.save(f'{dataset_path}/{dataset_name}/tiles_{tile_measure}/{dataset_type}/gt/{name}_{count}.tif')
            print(f'{name} DONE!\nTiles created:\t{count}\n')


# DEVO PROVARE A CALCOLARE LA MEDIA DEL VALORE DELLE MASCHERE PER CAPIRE SE SONO BILANCIATI I DATI
# QUINDI, FARE TIPO LA SOMMA SUI PIXEL DELLE MASCHERE E POI DIVIDERE PER LA DIMENSIONE DELLA TILE. SE IL RISULTATO È CIRCA 0.5
# ALLORA POSSO PENSARE CHE IL DATASET SIA BILANCIATO, CREDO

server_path = '/home/antoniocorvino/Projects/BuildingsExtraction/datasets'
nas_path = '/mnt/nas151/sar/Footprint/datasets'
local_path = '/Users/corvino/PycharmProjects/BuildingsExtraction/datasets'

massachusetts_dataset_name = 'MassachusettsBuildingsDataset'
aerial_dataset_name = 'InriaAerialDataset'
whu_dataset_name = 'WHUBuildingDataset'

tile_measure = 256

clear_tiles_directory(whu_dataset_name, nas_path, tile_measure)
tiles_creation(whu_dataset_name, nas_path, tile_measure, maps_to_use=1500)
