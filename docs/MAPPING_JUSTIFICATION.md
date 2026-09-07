# Why each MMFakeBench folder maps to its scenario

Source of truth: *MMFakeBench: A Mixed-Source Multimodal Misinformation
Detection Benchmark for LVLMs*, Liu et al., ICLR 2025 (arXiv 2406.08772v3),
section 3. Quotes are verbatim. The JSON evidence column is what every record
of that folder carries in `MMFakeBench_val.json` / `MMFakeBench_test.json`
(`fake_cls` / `text_source` / `image_source`), counted over all 11,000 records.

The five scenarios (project instructions):
1 real text + real image, out-of-context. 2 fake text + real image.
3 real text + fake image. 4 real text + real image, genuine. 5 fake text + fake image.

## Scenario 4, genuine (3,300)

Folders: `bbc`, `guardian`, `usa_today`, `wash`, `coco`, `fakeddit`
JSON: `original` / VisualNews, MS-COCO or Fakeddit / same as text.

> "we collect 3,300 real data pairs, ensuring both textual and visual veracity
> and exhibiting strong image-text consistency ... we construct the real
> dataset from the same corresponding sources, including MS-COCO, VisualNews,
> and real image-text pairs from Fakeddit. We further divide VisualNews into
> four distinct news sources: The Guardian, BBC, USA TODAY, and The Washington
> Post. Finally, we build the real dataset by equally selecting from six
> distinct sources." (3.2)

## Scenario 1, out-of-context (1,650)

Folders: `Newsclipings_person`, `Newsclipings_scene`, `Newsclipings_semantic`
JSON: `mismatch` / Newsclipings / Newsclipings.

> "In cross-modal consistency distortion, both the text and image with
> veracity, but either the text or image is replaced/manipulated to disrupt
> their overall consistency." (3.1.3)
> "Our dataset contains three types of repurposed inconsistency, curated
> directly from the NewsCLIPings dataset: semantic query, person query, and
> scene query." (3.1.3)

Both halves real, pairing wrong: scenario 1.

## Scenario 2, fake text + real image (1,925)

### `rumor_match`, `politicat_match`, `gossipcop_match` (825)
JSON: `textual_veracity_distortion` / Fakenewsnet or Gossipcop / Repurposed Image.

> "Natural Rumor. We select natural rumors from Politifact and Gossipcop"
> (3.1.1)
> "Repurposed Image: To avoid creating new high-risk images, especially for
> sensitive topics like politics and gossip, we use repurposed images from the
> VisualNews dataset, which contains numerous image-text pairs from real-world
> news sources. We select images with high semantic relevance to the textual
> rumors based on text-image CLIP similarity" (3.1.1)

Rumour caption, real photo taken from VisualNews: scenario 2.

### `chatgpt_match` (550)
JSON: `textual_veracity_distortion` / GPT-generated Rumor / Repurposed Image.

> "GPT-generated Rumor. We instruct ChatGPT (gpt-3.5-turbo) to produce rumors
> via three prompt approaches" (3.1.1), paired with a repurposed real image
> (same quote as above; `image_source` = "Repurposed Image").

### `DGM4_text_edit_senti` (550)
JSON: `mismatch` / DGM4 / DGM4.

> "For text editing, we select samples from the DGM4 dataset, which modifies
> sentiment words with their antonyms." (3.1.3)

Caption manipulated (antonym swap), photo untouched: scenario 2. This is the
same convention DGM4 itself uses (`text_attribute` = manipulated text).

## Scenario 3, real text + fake image (1,650)

> "The visual veracity distortion dataset comprises 1,100 samples where the
> text is real and the misinformation exists in the image." (3.1.2)

### `Fakeddit_photo_edit` (550)
JSON: `visual_veracity_distortion` / Fakeddit / Fakeddit.

> "PS-Edited Image. The PS-edited images are derived from the 'manipulated
> content' samples in the Fakeddit dataset ... ten of the volunteers
> participate in selecting 550 PS-edited images containing fact-conflicting
> content" (3.1.2)

### `antifact_image_generation` (550)
JSON: `visual_veracity_distortion` / MS-COCO, VisualNews or blank / AI-generated Image or blank.
(100 test records have blank source fields; the folder identifies them.)

> "AI-generated Image ... we first collect image-text pairs from the MS-COCO
> and VisualNews datasets. Based on the original text captions, we use ChatGPT
> to generate corresponding fact-conflicting descriptions ... The resulting
> text-image pairs contain original factual text and generated images with
> additional fact-conflicting information." (3.1.2)

### `coco_image_edit` (550)
JSON: `mismatch` / COCO-Counterfactuals / COCO-Counterfactuals.

> "we build upon the COCO-Counterfactuals dataset ... which encompasses
> original image-text pairs (text_ori, img_ori) and edited image-text pairs
> (text_edit, img_edit) which are obtained via Instruct-Pix2Pix model ... we
> reassemble the two pairs and obtain ... (text_ori, img_edit) as image-edited
> consistency distortion samples." (3.1.3)

Original caption, AI-edited image: scenario 3.

## Scenario 5, fake text + fake image (2,475)

### `fever_AI` (1,100)
JSON: `textual_veracity_distortion` / Fever / AI-generated Image.

> "Artificial Rumor. We collect artificial rumors from the FEVER dataset"
> (3.1.1)
> "AI-generated Image: For artificial rumors and their derived GPT-generated
> rumors, as well as some less harmful gossip, we utilize generative models to
> create supporting images. We utilize three advanced models: Stable Diffusion
> XL, DALL-E3, and Midjourney V6" (3.1.1)

### `llm_rewrite`, `llm_gossip_md_generation`, `llm_science_md_generation` (550)
JSON: `textual_veracity_distortion` / GPT-generated Rumor / AI-generated Image.

> "GPT-generated Rumor ... three prompt approaches: arbitrary generation,
> rewriting generation, and information manipulation." (3.1.1), paired with
> AI-generated images (quote above; `image_source` = "AI-generated Image").

### `gossipcop_midjourney` (275)
JSON: `textual_veracity_distortion` / Fakenewsnet or Gossipcop / AI-generated Image.

> "as well as some less harmful gossip, we utilize generative models to create
> supporting images" (3.1.1)

### `coco_text_edit` (550)
JSON: `mismatch` / COCO-Counterfactuals / COCO-Counterfactuals.

> "(text_edit, img_ori) as text-edited consistency distortion samples." (3.1.3)

Edited caption. The image is `img_ori` of a COCO-Counterfactuals pair, and in
that dataset every image is generated: "the two corresponding synthetic images
differ only in terms of the altered subject" (COCO-Counterfactuals, Le et al.,
NeurIPS 2023 Datasets and Benchmarks; HuggingFace `Intel/COCO-Counterfactuals`).
Edited caption + synthetic image: scenario 5.

**This is the one line that is a judgement call.** MMFakeBench treats
`img_ori` as the original of the pair; by the project's definition (is the
picture fake?) it is generated. If the supervisor prefers to follow
MMFakeBench's view, the folder moves to scenario 2 by changing one line of
`FOLDER_TO_GROUP`. N (1,650) does not change either way.

## Totals check

Counting folders by their `fake_cls` gives original 3,300 / textual 3,300 /
visual 1,100 / mismatch 3,300, matching "30% for textual veracity distortion,
10% for visual veracity distortion, 30% for cross-modal consistency
distortion, and 30% for real data" (3.3) over 11,000 pairs. The loader
recomputes this on every run and refuses to continue if it differs.
