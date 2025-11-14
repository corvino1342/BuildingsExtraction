from PIL import Image
import os
import shutil

def clear_tiles_directory(dataset_name):
    if os.path.exists(f'datasets/{dataset_name}/tiles'):
        print('Previous tiles erasing...')
        shutil.rmtree(f'datasets/{dataset_name}/tiles')
        print('DONE!')
    os.makedirs(f'datasets/{dataset_name}/tiles')

def save_tiles(img, msk, out_dir, img_name, tile_measure, gt):
    count = 0
    skipped = 0

    height, width = img.size



    for i in range(0, height, tile_measure):
        for j in range(0, width, tile_measure):
            count += 1

            box = (j, i, j + tile_measure, i+ tile_measure)
            mask_tile = msk.crop(box)

            if mask_tile.size != (tile_measure, tile_measure) or mask_tile.getextrema() == ((0,0),(0,0),(0,0)):
                skipped += 1
                continue
            image_tile = img.crop(box)
            image_tile.save(out_dir + f'/images/{img_name}_{count}.tif')
            if gt:
                mask_tile.save(out_dir + f'/gt/{img_name}_{count}.tif')
    print(f'{img_name} DONE! --- tiles skipped: {skipped}/{count} ({(100 * skipped/count):.1f}%)')

def tiles_creation(dataset_name, tile_measure, maps_to_use):

    os.makedirs(f'datasets/{dataset_name}/tiles', exist_ok=True)

    for dataset_type in ['train', 'val', 'test']:

        print(f'Dataset type ------- {dataset_type}')

        gt = True
        if not os.path.exists(f'datasets/{dataset_name}/{dataset_type}/gt'):
            print(f'---------{dataset_type} dataset has not Ground Truth---------')
            gt = False

        os.makedirs(f'datasets/{dataset_name}/tiles/{dataset_type}/', exist_ok=True)
        os.makedirs(f'datasets/{dataset_name}/tiles/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'datasets/{dataset_name}/tiles/{dataset_type}/gt', exist_ok=True)

        full_maps = sorted(os.path.splitext(f)[0] for f in os.listdir(f'datasets/{dataset_name}/{dataset_type}/images') if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg')))

        print(f'Maps used in {dataset_type}: {maps_to_use}/{len(full_maps)}..................')

        full_maps = full_maps[:maps_to_use]


        for name in full_maps:


            image = Image.open(f'datasets/{dataset_name}/{dataset_type}/images/{name}.tif')

            if gt:
                mask = Image.open(f'datasets/{dataset_name}/{dataset_type}/gt/{name}.tif')

            count = 0
            skipped = 0

            height, width = image.size

            for i in range(0, height, tile_measure):
                for j in range(0, width, tile_measure):
                    count += 1
                    box = (j, i, j + tile_measure, i + tile_measure)

                    if gt:
                        mask_tile = mask.crop(box)

                        if (mask_tile.size != (tile_measure, tile_measure) or
                            mask_tile.getextrema() == ((0, 0), (0, 0), (0, 0)) or
                            mask_tile.getextrema() == (0, 0)):                      # VERIFICARE COME MAI I DUE DATASET SALVANO IN MODO DIVERSO LE MASCHERE, UNO RGB E UNA BW

                            skipped += 1
                            continue
                    image_tile = image.crop(box)
                    image_tile.save(f'datasets/{dataset_name}/tiles/{dataset_type}/images/{name}_{count}.tif')
                    if gt:
                        mask_tile.save(f'datasets/{dataset_name}/tiles/{dataset_type}/gt/{name}_{count}.tif')
            print(f'{name} DONE! --- tiles skipped: {skipped}/{count} ({(100 * skipped / count):.1f}%)\n\n')


# DEVO PROVARE A CALCOLARE LA MEDIA DEL VALORE DELLE MASCHERE PER CAPIRE SE SONO BILANCIATI I DATI
# QUINDI, FARE TIPO LA SOMMA SUI PIXEL DELLE MASCHERE E POI DIVIDERE PER LA DIMENSIONE DELLA TILE. SE IL RISULTATO È CIRCA 0.5
# ALLORA POSSO PENSARE CHE IL DATASET SIA BILANCIATO, CREDO

dataset_name = 'MassachusettsBuildingsDataset'
dataset_name = 'AerialImageDataset'
clear_tiles_directory(dataset_name)
tiles_creation(dataset_name, tile_measure=256, maps_to_use=1)
