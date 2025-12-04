from PIL import Image
import os
import shutil


def clear_tiles_directory(dataset_name, dataset_path):
    if os.path.exists(f'{dataset_path}/{dataset_name}/tiles'):
        print('Previous tiles erasing...')
        shutil.rmtree(f'{dataset_path}/{dataset_name}/tiles')
        print('DONE!\n\n\n')
    os.makedirs(f'{dataset_path}/{dataset_name}/tiles')


def tiles_creation(dataset_name, dataset_path, tile_measure, maps_to_use):

    os.makedirs(f'{dataset_path}/{dataset_name}/tiles', exist_ok=True)

    for dataset_type in ['train', 'val', 'test']:

        print(f'Dataset type ------- {dataset_type}')

        gt = True
        if not os.path.exists(f'/home/antoniocorvino/Projects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/gt'):
            print(f'---------{dataset_type} dataset has not Ground Truth---------')
            gt = False

        os.makedirs(f'{dataset_path}/{dataset_name}/tiles/{dataset_type}/', exist_ok=True)
        os.makedirs(f'{dataset_path}/{dataset_name}/tiles/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'{dataset_path}/{dataset_name}/tiles/{dataset_type}/gt', exist_ok=True)

        full_maps = sorted(os.path.splitext(f)[0] for f in os.listdir(f'datasets/{dataset_name}/{dataset_type}/images') if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg')))

        print(f'Maps used in {dataset_type}: {maps_to_use}/{len(full_maps)}..................')

        full_maps = full_maps[:maps_to_use]


        for name in full_maps:
            image = Image.open(f'/home/antoniocorvino/Projects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/images/{name}.tif')
            if gt:
                mask = Image.open(f'/home/antoniocorvino/Projects/BuildingsExtraction/datasets/{dataset_name}/{dataset_type}/gt/{name}.tif')

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
                    image_tile.save(f'{dataset_path}/{dataset_name}/tiles/{dataset_type}/images/{name}_{count}.tif')
                    if gt:
                        mask_tile.save(f'{dataset_path}/{dataset_name}/tiles/{dataset_type}/gt/{name}_{count}.tif')
            print(f'{name} DONE!\n\n')


# DEVO PROVARE A CALCOLARE LA MEDIA DEL VALORE DELLE MASCHERE PER CAPIRE SE SONO BILANCIATI I DATI
# QUINDI, FARE TIPO LA SOMMA SUI PIXEL DELLE MASCHERE E POI DIVIDERE PER LA DIMENSIONE DELLA TILE. SE IL RISULTATO È CIRCA 0.5
# ALLORA POSSO PENSARE CHE IL DATASET SIA BILANCIATO, CREDO

server_path = '/home/antoniocorvino/Projects/BuildingsExtraction/datasets'
nas_path = '/mnt/nas151/sar/Footprint/datasets'

massachusetts_dataset_name = 'MassachusettsBuildingsDataset'
aerial_dataset_name = 'AerialImageDataset'

clear_tiles_directory(aerial_dataset_name, server_path)
tiles_creation(aerial_dataset_name, server_path, tile_measure=128, maps_to_use=1)
