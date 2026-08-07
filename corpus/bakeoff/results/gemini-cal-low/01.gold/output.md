the WWW as a network has boomed after it has been discovered that the degree distribution of the webpages follows a power-law over several orders of magnitude (Albert, Jeong, Barabási 1999, Kumar et al. 1999). Since the edges of the WWW are directed, the network is characterized by two degree distributions: the distribution of outgoing edges, $P_{out}(k)$, signifies the probability that a document has $k$ outgoing hyperlinks and the distribution of incoming edges, $P_{in}(k)$, is the probability that $k$ hyperlinks point to a certain document. Several studies have established that both $P_{out}(k)$ and $P_{in}(k)$ have power-law tails:

$$P_{out}(k) \sim k^{-\gamma_{out}} \quad \text{and} \quad P_{in}(k) \sim k^{-\gamma_{in}}. \tag{3}$$

WORLD-WIDE WEB
HOME PAGE

INTERNET

FIG. 1. Network structure of the World-Wide Web and the Internet. Upper panel: the nodes of the World-Wide Web are web documents, connected with directed hyperlinks (URLs). Lower panel: on the Internet the nodes are the routers and computers, the edges are the wires and cables that physically connect them. Figure courtesy of István Albert.

Albert, Jeong and Barabási (1999) have studied a subset of the WWW containing $325,729$ nodes and have found $\gamma_{out} = 2.45$ and $\gamma_{in} = 2.1$. Kumar et al. (1999) used a 40 million document crawl by Alexa Inc., obtaining $\gamma_{out} = 2.38$ and $\gamma_{in} = 2.1$ (see also Kleinberg et al. 1999). A later survey of the WWW topology by Broder et al. (2000) used two 1999 Altavista crawls containing in total 200 million documents, obtaining $\gamma_{out} = 2.72$ and $\gamma_{in} = 2.1$ with scaling holding close to five orders of magnitude (Fig. 2). Adamic and Huberman (2000) used a somewhat different representation of the WWW, each node representing a separate domain name and two nodes being connected if any of the pages in one domain linked to any page in the other. While this method lumps together often thousands of pages that are on the same domain, representing a nontrivial aggregation of the nodes, the distribution of incoming edges still followed a power-law with $\gamma_{in}^{dom} = 1.94$.

Note that $\gamma_{in}$ is the same for all measurements at the document level despite the two years time delay between the first and last web crawl, during which the WWW had grown at least five times larger. On the other hand, $\gamma_{out}$ has an increasing tendency with the sample size or time (see Table II).

Despite the large number of nodes, the WWW displays the small world property. This was first reported by Albert, Jeong and Barabási (1999), who found that the average path length for a sample of $325,729$ nodes was 11.2 and predicted, using finite size scaling, that for the full WWW of 800 million nodes that would be around 19. Subsequent measurements of Broder et al. (2000) found that the average path length between nodes in a 200 million sample of the WWW is 16, in agreement with the finite size prediction for a sample of this size. Finally, the domain level network displays an average path length of 3.1 (Adamic 1999).

The directed nature of the WWW does not allow us to measure the clustering coefficient using Eq. (1). One way to avoid this difficulty is to make the network undirected, making each edge bidirectional. This was the path followed by Adamic (1999) who studied the WWW at the domain level using an 1997 Alexa crawl of 50 million webpages distributed between $259,794$ sites. Adamic removed the nodes which have only one edge, focusing on a network of $153,127$ sites. While these modifications are expected to increase somewhat the clustering coefficient, she found $C = 0.1078$, orders of magnitude higher than $C_{rand} = 0.00023$ corresponding to a random graph of the same size and average degree.

$10^0$
$10^{-2}$
$10^{-4}$
$10^{-6}$
$10^{-8}$
$10^{-10}$
$P_{out}(k)$

$10^{-2}$ $10^0$ $10^2$ $10^4$ $10^6$
$k$
(a)

$P_{in}(k)$

$10^{-2}$ $10^0$ $10^2$ $10^4$ $10^6$
$k$
(b)

4

---

TABLE I. The general characteristics of several real networks. For each network we indicated the number of nodes, the average degree $\langle k \rangle$, the average path length $\ell$ and the clustering coefficient $C$. For a comparison we have included the average path length $\ell_{rand}$ and clustering coefficient $C_{rand}$ of a random graph with the same size and average degree. The last column identifies the symbols in Figs. 8 and 9.

| Network | Size | $\langle k \rangle$ | $\ell$ | $\ell_{rand}$ | $C$ | $C_{rand}$ | Reference | Nr. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| WWW, site level, undir. | $153,127$ | 35.21 | 3.1 | 3.35 | 0.1078 | 0.00023 | Adamic 1999 | 1 |
| Internet, domain level | 3015 - 6209 | 3.52 - 4.11 | 3.7 - 3.76 | 6.36 - 6.18 | 0.18 - 0.3 | 0.001 | Yook et al. 2001a, Pastor-Satorras et al. 2001 | 2 |
| Movie actors | $225,226$ | 61 | 3.65 | 2.99 | 0.79 | 0.00027 | Watts, Strogatz 1998 | 3 |
| LANL coauthorship | $52,909$ | 9.7 | 5.9 | 4.79 | 0.43 | $1.8 \times 10^{-4}$ | Newman 2001a,b | 4 |
| MEDLINE coauthorship | $1,520,251$ | 18.1 | 4.6 | 4.91 | 0.066 | $1.1 \times 10^{-5}$ | Newman 2001a,b | 5 |
| SPIRES coauthorship | $56,627$ | 173 | 4.0 | 2.12 | 0.726 | 0.003 | Newman 2001a,b,c | 6 |
| NCSTRL coauthorship | $11,994$ | 3.59 | 9.7 | 7.34 | 0.496 | $3 \times 10^{-4}$ | Newman 2001a,b | 7 |
| Math coauthorship | $70,975$ | 3.9 | 9.5 | 8.2 | 0.59 | $5.4 \times 10^{-5}$ | Barabási et al. 2001 | 8 |
| Neurosci. coauthorship | $209,293$ | 11.5 | 6 | 5.01 | 0.76 | $5.5 \times 10^{-5}$ | Barabási et al. 2001 | 9 |
| E. coli, substrate graph | 282 | 7.35 | 2.9 | 3.04 | 0.32 | 0.026 | Wagner, Fell 2000 | 10 |
| E. coli, reaction graph | 315 | 28.3 | 2.62 | 1.98 | 0.59 | 0.09 | Wagner, Fell 2000 | 11 |
| Ythan estuary food web | 134 | 8.7 | 2.43 | 2.26 | 0.22 | 0.06 | Montoya, Solé 2000 | 12 |
| Silwood park food web | 154 | 4.75 | 3.40 | 3.23 | 0.15 | 0.03 | Montoya, Solé 2000 | 13 |
| Words, cooccurence | 460.902 | 70.13 | 2.67 | 3.03 | 0.437 | 0.0001 | Cancho, Solé 2001 | 14 |
| Words, synonyms | $22,311$ | 13.48 | 4.5 | 3.84 | 0.7 | 0.0006 | Yook et al. 2001 | 15 |
| Power grid | $4,941$ | 2.67 | 18.7 | 12.4 | 0.08 | 0.005 | Watts, Strogatz 1998 | 16 |
| C. Elegans | 282 | 14 | 2.65 | 2.25 | 0.28 | 0.05 | Watts, Strogatz 1998 | 17 |

TABLE II. The scaling exponents characterizing the degree distribution of several scale-free networks, for which $P(k)$ follows a power-law (2). We indicate the size of the network, its average degree $\langle k \rangle$ and the cutoff $\kappa$ for the power-law scaling. For directed networks we list separately the indegree ($\gamma_{in}$) and outdegree ($\gamma_{out}$) exponents, while for the undirected networks, marked with a star, these values are identical. The columns $\ell_{real}$, $\ell_{rand}$ and $\ell_{pow}$ compare the average path length of real networks with power-law degree distribution and the prediction of random graph theory (17) and that of Newman, Strogatz and Watts (2000) (62), as discussed in Sect. V. The last column identifies the symbols in Figs. 8 and 9.

| Network | Size | $\langle k \rangle$ | $\kappa$ | $\gamma_{out}$ | $\gamma_{in}$ | $\ell_{real}$ | $\ell_{rand}$ | $\ell_{pow}$ | Reference | Nr. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| WWW | $325,729$ | 4.51 | 900 | 2.45 | 2.1 | 11.2 | 8.32 | 4.77 | Albert, Jeong, Barabási 1999 | 1 |
| WWW | $4 \times 10^7$ | 7 | | 2.38 | 2.1 | | | | Kumar et al. 1999 | 2 |
| WWW | $2 \times 10^8$ | 7.5 | $4,000$ | 2.72 | 2.1 | 16 | 8.85 | 7.61 | Broder et al. 2000 | 3 |
| WWW, site | $260,000$ | | | | 1.94 | | | | Huberman, Adamic 2000 | 4 |
| Internet, domain$*$ | $3,015 - 4,389$ | 3.42 - 3.76 | 30 - 40 | 2.1 - 2.2 | 2.1 - 2.2 | 4 | 6.3 | 5.2 | Faloutsos 1999 | 5 |
| Internet, router$*$ | $3,888$ | 2.57 | 30 | 2.48 | 2.48 | 12.15 | 8.75 | 7.67 | Faloutsos 1999 | 6 |
| Internet, router$*$ | $150,000$ | 2.66 | 60 | 2.4 | 2.4 | 11 | 12.8 | 7.47 | Govindan 2000 | 7 |
| Movie actors$*$ | $212,250$ | 28.78 | 900 | 2.3 | 2.3 | 4.54 | 3.65 | 4.01 | Barabási, Albert 1999 | 8 |
| Coauthors, SPIRES$*$ | $56,627$ | 173 | $1,100$ | 1.2 | 1.2 | 4 | 2.12 | 1.95 | Newman 2001b,c | 9 |
| Coauthors, neuro.$*$ | $209,293$ | 11.54 | 400 | 2.1 | 2.1 | 6 | 5.01 | 3.86 | Barabási et al. 2001 | 10 |
| Coauthors, math$*$ | $70,975$ | 3.9 | 120 | 2.5 | 2.5 | 9.5 | 8.2 | 6.53 | Barabási et al. 2001 | 11 |
| Sexual contacts$*$ | 2810 | | | 3.4 | 3.4 | | | | Liljeros et al. 2001 | 12 |
| Metabolic, E. coli | 778 | 7.4 | 110 | 2.2 | 2.2 | 3.2 | 3.32 | 2.89 | Jeong et al. 2000 | 13 |
| Protein, S. cerev.$*$ | 1870 | 2.39 | | 2.4 | 2.4 | | | | Mason et al. 2000 | 14 |
| Ythan estuary$*$ | 134 | 8.7 | 35 | 1.05 | 1.05 | 2.43 | 2.26 | 1.71 | Montoya, Solé 2000 | 14 |
| Silwood park$*$ | 154 | 4.75 | 27 | 1.13 | 1.13 | 3.4 | 3.23 | 2 | Montoya, Solé 2000 | 16 |
| Citation | $783,339$ | 8.57 | | | 3 | | | | Redner 1998 | 17 |
| Phone-call | $53 \times 10^6$ | 3.16 | | 2.1 | 2.1 | | | | Aiello et al. 2000 | 18 |
| Words, cooccurence$*$ | $460,902$ | 70.13 | | 2.7 | 2.7 | | | | Cancho, Solé 2001 | 19 |
| Words, synonyms$*$ | $22,311$ | 13.48 | | 2.8 | 2.8 | | | | Yook et al. 2001 | 20 |

8