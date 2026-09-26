"""Development-only coverage check for a possible small-face forensic extension."""
import json
import time

import numpy as np
from PIL import Image

from .development import OUT,DATASETS
from .diagnostic_groups import annotations
from .face_crop import FaceCrop,DIRECTORY
from .preprocess import read_rows,sha256


def main():
    rows=[r for r in read_rows(DATASETS['dgm4']) if r['split'] in ('train','val')]
    locator=FaceCrop();records=[];begin=time.monotonic()
    for i,row in enumerate(rows,1):
        with Image.open(row['image_path']) as image:found=locator.locate(image)
        record=dict(sample_id=row['sample_id'],split=row['split'],label=int(row['label']),**found)
        if int(row['label']):
            key='DGM4/'+row['image_path'].split('/DGM4/',1)[1]
            target=annotations(row['source_metadata'])[key]['fake_image_box']
            left,top,right,bottom=found['box'];x1,y1,x2,y2=target
            area=max(0,min(right,x2)-max(left,x1))*max(0,min(bottom,y2)-max(top,y1))
            record['annotated_face_fraction_retained']=area/((x2-x1)*(y2-y1))
            record['equivalent_face_side_after_crop_resize']=224*np.sqrt(area/((right-left)*(bottom-top)))
            record['equivalent_face_side_after_full_resize']=224*np.sqrt((x2-x1)*(y2-y1)/(int(row['width'])*int(row['height'])))
            fractions=[]
            for left,top,right,bottom in found['boxes']:
                area=max(0,min(right,x2)-max(left,x1))*max(0,min(bottom,y2)-max(top,y1))
                fractions.append(area/((x2-x1)*(y2-y1)))
            record['face_fraction_present_among_regions']={str(k):max(fractions[:k]) for k in (1,2,3,5,100)}
        records.append(record)
        if i%500==0:print(i,'/',len(rows),'seconds',round(time.monotonic()-begin),flush=True)
    val=[r for r in records if r['split']=='val'];fake=[r for r in val if r['label']]
    report=dict(n=len(records),validation_n=len(val),validation_detected=sum(r['detected'] for r in val),
                manipulated_validation_n=len(fake),manipulated_detected=sum(r['detected'] for r in fake),
                annotated_face_completely_outside_crop=sum(r['annotated_face_fraction_retained']==0 for r in fake),
                annotated_face_less_than_half_in_crop=sum(r['annotated_face_fraction_retained']<.5 for r in fake),
                median_equivalent_face_pixels_before=float(np.median([r['equivalent_face_side_after_full_resize'] for r in fake])),
                median_equivalent_face_pixels_after=float(np.median([r['equivalent_face_side_after_crop_resize'] for r in fake])),
                locator_source=json.loads((DIRECTORY/'source.json').read_text()),dataset_sha256=sha256(DATASETS['dgm4']),
                input='Only pixels are supplied to the face locator. Annotation boxes are read afterward for this coverage diagnostic.',
                classifier_trained=False,final_test_used=False)
    report['regions_per_image_mean']=float(np.mean([len(r['boxes']) for r in val]))
    report['edited_region_coverage_by_largest_k_faces']={str(k):dict(outside_all_regions=sum(r['face_fraction_present_among_regions'][str(k)]==0 for r in fake),less_than_half_visible=sum(r['face_fraction_present_among_regions'][str(k)]<.5 for r in fake)) for k in (1,2,3,5,100)}
    (OUT/'face_development_crops.json').write_text(json.dumps(records,indent=2))
    (OUT/'face_feasibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
