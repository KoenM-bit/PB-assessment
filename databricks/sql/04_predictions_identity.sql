-- Migration for catalogs created before address/property_key columns existed.
-- Safe to skip if 03_gold.sql already created predictions with these columns.
-- Re-running may error with COLUMN_ALREADY_EXISTS — that is expected.

ALTER TABLE ${catalog}.gold.predictions ADD COLUMN address STRING;
ALTER TABLE ${catalog}.gold.predictions ADD COLUMN postcode STRING;
ALTER TABLE ${catalog}.gold.predictions ADD COLUMN property_key STRING;
