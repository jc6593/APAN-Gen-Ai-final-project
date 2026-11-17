"""Thin CLI adapter for the isolated Evidence-first workflow."""

from pathlib import Path


DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def run_mode(
    mode,
    project_root,
    taxonomy,
    qwen_model_id,
    limit=0,
    show=5,
    force=False,
    max_retries=1,
    nli_model_id=DEFAULT_NLI_MODEL,
    nli_device="auto",
    nli_batch_size=8,
    force_claims=False,
    product_config=None,
    catalog_config=None,
    source_catalog_config=None,
    demo_config=None,
):
    """Dispatch a new mode without importing optional ML dependencies early."""
    root = Path(project_root).resolve()
    laptop_product_modes = {"laptop-pilot", "laptop-summary"}
    if product_config is not None and mode not in laptop_product_modes:
        raise ValueError(
            "--product-config is available only for laptop modes "
            "laptop-pilot and laptop-summary"
        )
    catalog_modes = {"laptop-catalog-expand", "laptop-catalog-promote"}
    if catalog_config is not None and mode not in catalog_modes:
        raise ValueError(
            "--catalog-config is available only for catalog expansion or promotion"
        )
    if source_catalog_config is not None and mode != "laptop-catalog-promote":
        raise ValueError(
            "--source-catalog-config is available only for laptop-catalog-promote"
        )
    if demo_config is not None and mode != "laptop-demo-precompute":
        raise ValueError("--demo-config is available only for laptop-demo-precompute")
    if mode == "laptop-demo-precompute":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is not available for laptop-demo-precompute")
        if limit:
            raise ValueError("--limit is fixed by the featured demo config")
        if (
            product_config is not None
            or catalog_config is not None
            or source_catalog_config is not None
        ):
            raise ValueError(
                "--product-config and --catalog-config are not used with "
                "laptop-demo-precompute"
            )
        if demo_config is None:
            raise ValueError("--demo-config is required for laptop-demo-precompute")
        from .laptop_batch import run_featured_precompute

        return run_featured_precompute(
            project_root=root,
            demo_config=demo_config,
            show=show,
        )
    if mode == "laptop-catalog-expand":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is not available for laptop-catalog-expand")
        if limit:
            raise ValueError("--limit is not available for laptop-catalog-expand")
        if product_config is not None:
            raise ValueError("--product-config is not used with laptop-catalog-expand")
        if catalog_config is None:
            raise ValueError("--catalog-config is required for laptop-catalog-expand")
        from .catalog_collection import run_laptop_catalog_expansion

        return run_laptop_catalog_expansion(
            project_root=root,
            catalog_config=catalog_config,
        )
    if mode == "laptop-catalog-promote":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is not available for laptop-catalog-promote")
        if limit:
            raise ValueError("--limit is not available for laptop-catalog-promote")
        if product_config is not None or demo_config is not None:
            raise ValueError(
                "--product-config and --demo-config are not used with "
                "laptop-catalog-promote"
            )
        if catalog_config is None:
            raise ValueError("--catalog-config is required for laptop-catalog-promote")
        if source_catalog_config is None:
            raise ValueError(
                "--source-catalog-config is required for laptop-catalog-promote"
            )
        from .catalog_collection import promote_laptop_catalog

        return promote_laptop_catalog(
            project_root=root,
            source_catalog_config=source_catalog_config,
            target_catalog_config=catalog_config,
        )
    if mode == "laptop-catalog":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is not available for laptop-catalog")
        if limit:
            raise ValueError("--limit is not available for laptop-catalog")
        if product_config is not None:
            raise ValueError("--product-config is not used with laptop-catalog")
        from .laptop_products import list_laptop_products

        products = list_laptop_products(root)
        print("\n=== REVIEWLENS LAPTOP PRODUCT CATALOG ===")
        for row in products:
            print(
                f"  {row['product_id']}: {row['display_name']} "
                f"({row['available_review_count']} reviews; "
                f"scope={row['catalog_scope']})"
            )
            print(f"    config: {row['config_path']}")
        print("No models were loaded or called.")
        return products
    if mode in {"laptop-pilot", "laptop-summary"}:
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if product_config is None:
            raise ValueError(f"--product-config is required for {mode}")
        if force:
            raise ValueError(f"--force is not available for {mode}")
        laptop_limit = 10 if limit == 0 else limit
        if laptop_limit < 1:
            raise ValueError("Laptop product --limit must be at least 1")
        if mode == "laptop-pilot":
            from .laptop_pipeline import run_laptop_pilot

            return run_laptop_pilot(
                project_root=root,
                product_config=product_config,
                limit=laptop_limit,
                show=show,
                l2_model_id=qwen_model_id,
                max_retries=max_retries,
            )
        from .laptop_pipeline import summarize_laptop_run

        return summarize_laptop_run(
            project_root=root,
            product_config=product_config,
            limit=laptop_limit,
            show=show,
        )
    if mode == "ef-pilot":
        from .pipeline import run_pilot

        pilot_limit = 10 if limit == 0 else limit
        if not 1 <= pilot_limit <= 10:
            raise ValueError("Evidence-first pilot --limit must be between 1 and 10")
        return run_pilot(
            project_root=root,
            taxonomy=taxonomy,
            model_id=qwen_model_id,
            limit=pilot_limit,
            force=force,
            force_claims=force_claims,
            max_retries=max_retries,
            show=show,
        )
    if mode == "ef-nli":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        from .nli import run_nli

        return run_nli(
            project_root=root,
            model_id=nli_model_id,
            device=nli_device,
            batch_size=nli_batch_size,
            force=force,
            show=show,
        )
    if mode == "ef-summary":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        from .evaluation import summarize

        return summarize(project_root=root, show=show)
    if mode == "ef-eval":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is not available for ef-eval")
        if limit:
            raise ValueError("--limit is not available for ef-eval")
        from .evaluation_v2 import summarize_evaluation

        return summarize_evaluation(
            project_root=root,
            taxonomy=taxonomy,
            cohort="pilot",
            show=show,
        )
    if mode == "ef-holdout-prepare":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError(
                "--force cannot overwrite an immutable holdout selection"
            )
        if limit:
            raise ValueError("--limit is not available for ef-holdout-prepare")
        from .holdout import prepare_holdout

        return prepare_holdout(
            project_root=root,
            taxonomy=taxonomy,
            model_id=qwen_model_id,
        )
    if mode == "ef-holdout":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        from .holdout import run_holdout

        holdout_limit = 25 if limit == 0 else limit
        if not 1 <= holdout_limit <= 25:
            raise ValueError(
                "Evidence-first holdout --limit must be between 1 and 25"
            )
        return run_holdout(
            project_root=root,
            taxonomy=taxonomy,
            model_id=qwen_model_id,
            limit=holdout_limit,
            force=force,
            max_retries=max_retries,
            show=show,
        )
    if mode == "ef-holdout-summary":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is not available for ef-holdout-summary")
        if limit:
            raise ValueError("--limit is not available for ef-holdout-summary")
        from .evaluation_v2 import summarize_evaluation

        return summarize_evaluation(
            project_root=root,
            taxonomy=taxonomy,
            cohort="holdout",
            show=show,
        )
    if mode == "ef-v15-dev-prepare":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError(
                "--force cannot overwrite the immutable v1.5 dev selection"
            )
        if limit:
            raise ValueError("--limit is not available for ef-v15-dev-prepare")
        from .v15_dev import prepare_v15_dev

        return prepare_v15_dev(project_root=root)
    if mode == "ef-v15-dev":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        from .v15_dev import run_v15_dev

        dev_limit = 10 if limit == 0 else limit
        if not 1 <= dev_limit <= 20:
            raise ValueError(
                "Evidence-first v1.5 dev --limit must be between 1 and 20"
            )
        return run_v15_dev(
            project_root=root,
            taxonomy=taxonomy,
            model_id=qwen_model_id,
            limit=dev_limit,
            force=force,
            max_retries=max_retries,
            show=show,
        )
    if mode == "ef-v15-final-prepare":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError(
                "--force cannot overwrite the immutable v1.5 final selection"
            )
        if limit:
            raise ValueError("--limit is not available for ef-v15-final-prepare")
        if max_retries != 1:
            raise ValueError("The locked v1.5 final protocol requires --max-retries 1")
        from .v15_final import prepare_v15_final

        return prepare_v15_final(project_root=root)
    if mode == "ef-v15-final":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is forbidden for the locked v1.5 final run")
        if limit:
            raise ValueError(
                "--limit is forbidden for the locked v1.5 final run; all 15 are fixed"
            )
        if max_retries != 1:
            raise ValueError("The locked v1.5 final protocol requires --max-retries 1")
        from .v15_final import run_v15_final

        return run_v15_final(
            project_root=root,
            taxonomy=taxonomy,
            model_id=qwen_model_id,
        )
    if mode == "ef-v15-final-eval":
        if force_claims:
            raise ValueError("force_claims is available only for ef-pilot")
        if force:
            raise ValueError("--force is not available for ef-v15-final-eval")
        if limit:
            raise ValueError("--limit is not available for ef-v15-final-eval")
        from .v15_final_eval import evaluate_v15_final

        return evaluate_v15_final(
            project_root=root,
            taxonomy=taxonomy,
            show=show,
        )
    raise ValueError(f"Unknown Evidence-first mode: {mode}")
