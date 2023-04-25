def ndc_to_act():
    """
    Using athena mappings, we class each ndc to a ATC, using a Rxnorm mapping.
    The topic has been discussed here:
    https://forums.ohdsi.org/t/mapping-ndc-codes-to-higher-level-drug-classes/3179/5
    """
    # TODO: Unfinished business in favor of https://github.com/fabkury/ndc_map mapping
    query = """
    select 
        atc.concept_id as atc_id, atc.concept_name as atc_name, atc.concept_code as atc_code, atc.concept_class_id as atc_class,
        ndc.concept_id as ndc_id, ndc.concept_name as ndc_name, ndc.concept_code as ndc_code, ndc.concept_class_id as ndc_class
    from concept atc
    join concept_ancestor a on a.ancestor_concept_id=atc.concept_id
    join concept_relationship r on r.concept_id_1=a.descendant_concept_id and r.invalid_reason is null and r.relationship_id='Mapped from'
    join concept ndc on ndc.concept_id=concept_id_2 and ndc.vocabulary_id='NDC'
    where atc.vocabulary_id='ATC'
    """
