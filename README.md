SmartScrape is a formal
framework and reference architecture for machine learningbased WIE that unifies (i) a formal model of web pages,
targets, and selectors, (ii) a learn-validate-repair training loop
with measurable generalization guarantees, and (iii) a modular
system design that combines DOM, visual cues, text semantics,
and background ontologies. SmartScrape treats extraction as
a typed mapping from page elements to structured records
governed by a schema and integrity constraints. We introduce a
class of schema-constrained selectors that can be learned from
weak supervision, provide identifiability conditions for consistent
learning under site drift, and derive practical sample-efficiency
bounds for active labeling strategies. A prototype implementation
integrates with layout-analysis pipelines and supports both singlepage and site-wide training. In controlled experiments over
product, job, and event domains, SmartScrape reduces wrapper
breakage by 30–55% relative to strong baselines while maintaining
high precision. We further show how the formal layer enables
proof-carrying extraction: every output record comes with a
machine-checkable justification linked to constraints and learned
models, facilitating debugging and compliance. We conclude with
limitations—multimodal pages with heavy scripting and antibot defenses—and outline future work on retrieval-augmented
extraction and cross-site transfer.
