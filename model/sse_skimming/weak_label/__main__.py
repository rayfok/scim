from weak_label.labeling_utils import export_labeled_dataset, load_dataset
from weak_label.lfs import Annotator
from weak_label.types import Label


def main():
    dataset = load_dataset(src="output/dataset/raw/")

    ann = Annotator()
    lfs = [
        ann.contains_method_statement,
        ann.contains_result_statement,
        ann.contains_novelty_statement,
        # ann.exceeds_min_length,
        ann.exceeds_max_length,
        ann.is_in_method_section,
        ann.is_in_acknowledgements,
        ann.contains_link,
        ann.contains_arxiv,
        ann.is_list_elem_in_introduction,
        ann.is_nonauthor_background,
    ]

    labels = ann.apply_labeling_functions(dataset, lfs)
    lf_summary = ann.get_lf_summary(labels, lfs)
    print(lf_summary)

    preds = ann.predict(labels, agg_model="label")
    for pred, x in zip(preds, dataset):
        x.label = int(pred)

    export_labeled_dataset(
        dataset,
        text_and_labels_only=True,
        filter_abstain=False,
        dst="output/snorkel/weak_labels.json",
    )


if __name__ == "__main__":
    main()
