# Data Layout

Dataset files are not included in this repository. Put them under this
directory or pass another directory with `--data-root`.

## DHG Datasets

The following datasets are loaded through `dhg.data` and are cached by DHG:

```text
Cora
Citeseer
Pubmed
Cora-CA
DBLP-CA
Walmart
```

Their aliases are mapped internally to:

```text
CocitationCora
CocitationCiteseer
CocitationPubmed
CoauthorshipCora
CoauthorshipDBLP
WalmartTrips
```

## LE-Style Datasets

Use the following layout:

```text
data/
  zoo/
    zoo.content
    zoo.edges
  NTU2012/
    NTU2012.content
    NTU2012.edges
  ModelNet40/
    ModelNet40.content
    ModelNet40.edges
```

The loader also accepts `data/Zoo/Zoo.content` and `data/Zoo/Zoo.edges`.

## Legislative Datasets

Use the Cornell-style layout:

```text
data/
  senate-committees/
    node-labels-senate-committees.txt
    hyperedges-senate-committees.txt
  house-committees/
    node-labels-house-committees.txt
    hyperedges-house-committees.txt
```

