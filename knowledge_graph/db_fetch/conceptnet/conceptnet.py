import sqlite3


DB_PATH = "./KnowledgeGraph/db_fetch/conceptnet/conceptnet_normalized.db"

class Conceptnet_db:

    def __init__(self):
        self.db_path = DB_PATH
        self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.cursor = self.conn.cursor()
        print(f"Connected to ConceptNet DB at: {self.db_path}")

    def __del__(self):
        """Close the database connection when the object is destroyed."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def get_distinct_relations(self, concept: str, lang: str = "en") -> list[str]:
        """
        Get all distinct outgoing relation URLs for a given ConceptNet concept.

        Args:
            concept (str): Concept name (e.g., "dog")
            lang (str): Language code (default: "en")

        Returns:
            List[str]: Distinct ConceptNet relation URLs
        """
        concept = concept.strip().lower()
        node_url = f"http://conceptnet.io/c/{lang}/{concept}"



        query = """
        SELECT DISTINCT
            r.rel_url
        FROM edge_norm e
        JOIN node_norm n_start ON e.start_fk = n_start.node_pk
        JOIN rel_norm r ON e.rel_fk = r.rel_pk
        WHERE n_start.node_url = ?
        ORDER BY r.rel_url;
        """

        self.cursor.execute(query, (node_url,))
        relations = [row[0] for row in self.cursor.fetchall()]

        return relations


    # subclasses basically end nodes for isA relations
    def get_subclasses(self, target_concept: str, lang: str = "en") -> list[str]:   
        """
        Given a target concept, return all source node URLs
        that have an IsA relation pointing to it.

        Example:
            get_isA_sources("animal")
            -> ["http://conceptnet.io/c/en/dog", "http://conceptnet.io/c/en/cat", ...]

        Args:
            target_concept (str): Concept name (e.g., "animal")
            lang (str): Language code (default: "en")

        Returns:
            List[str]: Source ConceptNet node URLs
        """
        target_concept = target_concept.strip().lower()
        target_node_url = f"http://conceptnet.io/c/{lang}/{target_concept}"
        isa_rel_url = "http://conceptnet.io/r/IsA"


        query = """
        SELECT DISTINCT
            n_start.node_url,
            e.weight
        FROM edge_norm e
        JOIN node_norm n_start ON e.start_fk = n_start.node_pk
        JOIN node_norm n_end   ON e.end_fk = n_end.node_pk
        JOIN rel_norm r        ON e.rel_fk = r.rel_pk
        WHERE
            n_end.node_url = ?
            AND r.rel_url = ?
        ORDER BY e.weight DESC;
        """

        self.cursor.execute(query, (target_node_url, isa_rel_url))
        sources = [row[0] for row in self.cursor.fetchall()]

        # strip off the url, only keep the concept part
        sources = [url.split(f"/c/{lang}/")[1] for url in sources]

        # if the concept ends with /n, remove the \n part
        sources = [s.split("/")[0] for s in sources]

        # create a set to remove duplicates, then convert back to list
        sources = list(set(sources))


        return sources


    # superclasses basically start nodes for isA relations
    def get_superclasses(
        self,
        start_concept: str,
        lang: str = "en"
    ) -> list[tuple[str, float]]:
        """
        Given a start concept, return all end node URLs connected via an
        IsA relation, ordered by relationship weight (descending).

        Example:
            get_isa_targets("dog")
            -> [
                ("http://conceptnet.io/c/en/animal", 3.2),
                ("http://conceptnet.io/c/en/pet", 2.8),
                ...
            ]

        Args:
            start_concept (str): Concept name (e.g., "dog")
            lang (str): Language code (default: "en")

        Returns:
            List[Tuple[str, float]]: (end_node_url, weight) pairs
        """
        start_concept = start_concept.strip().lower()
        start_node_url = f"http://conceptnet.io/c/{lang}/{start_concept}"
        isa_rel_url = "http://conceptnet.io/r/IsA"


        query = """
        SELECT
            n_end.node_url,
            e.weight
        FROM edge_norm e
        JOIN node_norm n_start ON e.start_fk = n_start.node_pk
        JOIN node_norm n_end   ON e.end_fk = n_end.node_pk
        JOIN rel_norm r        ON e.rel_fk = r.rel_pk
        WHERE
            n_start.node_url = ?
            AND r.rel_url = ?
        ORDER BY e.weight DESC;
        """

        self.cursor.execute(query, (start_node_url, isa_rel_url))
        results = self.cursor.fetchall()

        # now, create list for only the end url concepts, we don't need the weights here
        end_urls = [row[0] for row in results]

        concepts = [url.split(f"/c/{lang}/")[1] for url in end_urls]


        # if the concept ends with /n, remove the \n part
        concepts = [s.split("/")[0] for s in concepts]

        return concepts

        
    # the end nodes for HasProperty relations
    def get_properties(self, start_concept: str, lang: str = "en") -> list[str]:
        """
        Given a start concept, return all end node URLs
        connected via a HasProperty relation.

        Example:
            get_hasproperty_targets("dog")
            -> ["http://conceptnet.io/c/en/furry", "http://conceptnet.io/c/en/loyal", ...]

        Args:
            start_concept (str): Concept name (e.g., "dog")
            lang (str): Language code (default: "en")

        Returns:
            List[str]: End ConceptNet node URLs
        """
        start_concept = start_concept.strip().lower()
        start_node_url = f"http://conceptnet.io/c/{lang}/{start_concept}"
        hasproperty_rel_url = "http://conceptnet.io/r/HasProperty"


        query = """
            SELECT
                n_end.node_url,
                e.weight
            FROM edge_norm e
            JOIN node_norm n_start ON e.start_fk = n_start.node_pk
            JOIN node_norm n_end   ON e.end_fk = n_end.node_pk
            JOIN rel_norm r        ON e.rel_fk = r.rel_pk
            WHERE
                n_start.node_url = ?
                AND r.rel_url = ?
            ORDER BY e.weight DESC;
        """

        self.cursor.execute(query, (start_node_url, hasproperty_rel_url))
        targets = [row[0] for row in self.cursor.fetchall()]

        # strip off the url, only keep the concept part
        targets = [url.split(f"/c/{lang}/")[1] for url in targets]

        return targets






