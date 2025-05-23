import datetime
import random
import re
import xml.etree.ElementTree as ET


class DecisionResult:
    """
    A class representing the result of a decision made by an LLM for a specific test case.

    Attributes:
        MODEL (str): The name of the LLM used to make the decision.
        TEMPERATURE (float): The LLM temperature parameter used to generate the decisions.
        SEED (int): The LLM seed used to generate the decisions.
        TIMESTAMP (str): The timestamp when the decision was made.
        CONTROL_OPTIONS (list[str]): A list containing the non-shuffled and non-reversed options available for the control template.
        CONTROL_OPTION_SHUFFLING (list[int]): A list containing the zero-based IDs (original positions before shuffling) of the shuffled or reversed options in the control template.
        CONTROL_ANSWER (str): The raw decision output from the deciding LLM for the control template.
        CONTROL_EXTRACTION (str): The final extracted decision by the deciding LLM for the control template.
        CONTROL_DECISION (int): The decision made by the LLM for the control template, corresponding to the zero-based index of the option in the non-shuffled and non-reversed control template.
        TREATMENT_OPTIONS (list[str]): A list containing the non-shuffled and non-reversed options available for the treatment template.
        TREATMENT_OPTION_SHUFFLING (list[int]): A list containing the zero-based IDs (original positions before shuffling) of the shuffled or reversed options in the treatment template.
        TREATMENT_ANSWER (str): The raw decision output from the deciding LLM for the treatment template.
        TREATMENT_EXTRACTION (str): The final extracted decision by the deciding LLM for the treatment template.
        TREATMENT_DECISION (int): The decision made by the LLM for the treatment template, corresponding to the zero-based index of the option in the non-shuffled and non-reversed treatment template.
        STATUS (str): The status of the decision, either 'OK' or 'ERROR'.
        ERROR_MESSAGE (str): The error message if the decision status is 'ERROR', else None.
    """

    def __init__(
        self,
        id: int,
        bias: str,
        model: str,
        control_options: list[str],
        control_option_order: list[int],
        control_answer: str,
        control_extraction: str,
        control_decision: int,
        treatment_options: list[str],
        treatment_option_order: list[int],
        treatment_answer: str,
        treatment_extraction: str,
        treatment_decision: int,
        temperature: float = None,
        seed: int = None,
        status: str = "OK",
        error_message: str = None,
    ):
        """
        Instantiates a new DecisionResult object.

        Args:
            model (str): The name of the LLM used to make the decision.
            control_options (list[str]): A list containing the (shuffled or reversed) options available for the control template.
            control_option_order (list[int]): A list containing the zero-based IDs (original positions before shuffling) of the options in the control template.
            control_answer (str): The raw decision output from the deciding LLM for the control template.
            control_extraction (str): The final extracted decision by the deciding LLM for the control template.
            control_decision (int): The decision made by the LLM for the control template, corresponding to the position in the shuffled or reversed options list with one-based indexing.
            treatment_options (list[str]): A list containing the (shuffled or reversed) options available for the treatment template.
            treatment_option_order (list[int]): A list containing the zero-based IDs (original positions before shuffling) of the options in the treatment template.
            treatment_answer (str): The raw decision output from the deciding LLM for the treatment template.
            treatment_extraction (str): The final extracted decision by the deciding LLM for the treatment template.
            treatment_decision (int): The decision made by the LLM for the treatment template, corresponding to the position in the shuffled or reversed options list with one-based indexing.
            temperature (float): The LLM temperature parameter used to generate the decisions.
            seed (int): The LLM seed used to generate the decisions.
            status (str): The status of the decision, either 'OK' or 'ERROR'.
            error_message (str): The error message if the decision status is 'ERROR'.
        """

        self.ID: int = id
        self.BIAS: str = bias
        self.MODEL: str = model
        self.TEMPERATURE: float = temperature
        self.SEED: int = seed
        self.TIMESTAMP: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Unshuffle the options and convert the decision to zero-based indexing
        if (
            control_options is not None
            and control_option_order is not None
            and control_decision is not None
        ):
            control_options, _, control_decision = self._unshuffle(
                control_options, control_option_order, control_decision, convert_to_zero_based=True
            )
        if (
            treatment_options is not None
            and treatment_option_order is not None
            and treatment_decision is not None
        ):
            treatment_options, _, treatment_decision = self._unshuffle(
                treatment_options,
                treatment_option_order,
                treatment_decision,
                convert_to_zero_based=True,
            )

        self.CONTROL_OPTIONS: list[str] = control_options
        self.CONTROL_OPTION_SHUFFLING: list[int] = control_option_order
        self.CONTROL_ANSWER: str = control_answer
        self.CONTROL_EXTRACTION: str = control_extraction
        self.CONTROL_DECISION: int = control_decision

        self.TREATMENT_OPTIONS: list[str] = treatment_options
        self.TREATMENT_OPTION_SHUFFLING: list[int] = treatment_option_order
        self.TREATMENT_ANSWER: str = treatment_answer
        self.TREATMENT_EXTRACTION: str = treatment_extraction
        self.TREATMENT_DECISION: int = treatment_decision

        self.STATUS: str = status
        self.ERROR_MESSAGE: str = error_message

    def _unshuffle(
        self,
        options: list[str],
        option_order: list[int],
        decision: int,
        convert_to_zero_based: bool = True,
    ) -> tuple[list[str], list[int], int]:
        """
        Undoes the shuffling or reversal of options and the final chosen decision.

        Args:
            options (list[str]): A list containing the shuffled (or reversed) options.
            option_order (list[int]): A list containing the zero-based IDs (original positions before shuffling) of the options in the template.
            decision (int): The decision made by the LLM, corresponding to the formatted, one-based ID of the option in the shuffled (or reversed) template.
            convert_to_zero_based (bool): Whether to convert the decision to zero-based indexing, i.e., first option has ID 0 instead of 1.

        Returns:
            tuple[list[str], list[int], int]: A tuple containing the unshuffled options, the unshuffled option order, and the unshuffled decision.
        """

        # Verify that options and option_order have the same length
        if len(options) != len(option_order):
            raise ValueError(
                f"Passed options and option order must have the same length. Found len(options)={len(options)} and len(option_order)={len(option_order)}."
            )

        # Verify that option order is a permutation of [0, 1, ..., len(options)-1]
        if set(option_order) != set(range(len(option_order))):
            raise ValueError(
                f"Passed option order must be a permutation of [0, 1, ..., len(options)-1]. Found {option_order}."
            )

        # Verify that the decision is within the range [1, len(options)]
        if decision < 1 or decision > len(options):
            raise ValueError(
                f"Passed decision must be within the range [1, {len(options)}]. Found {decision}."
            )

        # Unshuffle the options list and option order list based on the option order list
        unshuffled_options: list[str] = []
        unshuffled_option_order: list[int] = []
        for original_index in range(len(option_order)):
            shuffled_index = option_order.index(original_index)
            unshuffled_options.append(options[shuffled_index])
            unshuffled_option_order.append(option_order[shuffled_index])

        # Convert the decision into zero-based indexing
        decision -= 1

        # Unshuffle the decision
        unshuffled_decision: int = option_order[decision]

        # If requested, convert the unshuffled decision back to one-based indexing
        if not convert_to_zero_based:
            unshuffled_decision += 1

        return unshuffled_options, unshuffled_option_order, unshuffled_decision

    def __str__(self) -> str:
        return f"---DecisionResult---\n\nTIMESTAMP: {self.TIMESTAMP}\nMODEL: {self.MODEL}\nTEMPERATURE: {self.TEMPERATURE}\nSEED: {self.SEED}\n\nCONTROL OPTIONS: {self.CONTROL_OPTIONS}\nCONTROL OPTION SHUFFLING: {self.CONTROL_OPTION_SHUFFLING}\nRAW CONTROL ANSWER: {self.CONTROL_ANSWER}\nCONTROL EXTRACTION: {self.CONTROL_EXTRACTION}\nCONTROL DECISION: {self.CONTROL_DECISION}\n\nTREATMENT OPTIONS: {self.TREATMENT_OPTIONS}\nTREATMENT OPTION SHUFFLING: {self.TREATMENT_OPTION_SHUFFLING}\nRAW TREATMENT ANSWER: {self.TREATMENT_ANSWER}\nTREATMENT EXTRACTION: {self.TREATMENT_EXTRACTION}\nTREATMENT DECISION: {self.TREATMENT_DECISION}\nSTATUS: {self.STATUS}\nERROR MESSAGE: {self.ERROR_MESSAGE}\n\n------"

    def __repr__(self) -> str:
        return self.__str__()


class Insertion:
    """
    A class representing a text that has been inserted into a blank/gap in a Template object.

    Attributes:
        pattern (str): The pattern that was replaced by the insertion text.
        text (str): The text that was inserted into the blank/gap.
        origin (str): The origin of the text, either 'user' or 'model'.
    """

    def __init__(self, pattern: str, text: str, origin: str):
        self.pattern = pattern
        self.text = text
        self.origin = origin

    def __str__(self) -> str:
        origin_str = "None" if self.origin is None else f'"{self.origin}"'
        return f'Insertion(pattern="{self.pattern}", text="{self.text}", origin={origin_str})'

    def __repr__(self) -> str:
        return self.__str__()


class Template:
    """
    A class representing a single template (e.g., control or treatment template) for a cognitive bias test case.
    Uses xml.etree.ElementTree internally to store and manipulate the template contents.

    Attributes:
        _data (xml.etree.ElementTree.Element): An Element object storing the contents of this template.
    """

    def __init__(
        self,
        from_string: str = None,
        from_file: str = None,
        from_element: ET.Element = None,
        type: str = None,
    ):
        """
        Instantiates a new Template object. Up to one source (string, file, or element) can be provided. If no source is provided, an empty template will be created.

        Args:
            from_string (str): The XML-like string from which to parse the template.
            from_file (str): The path of the XML file from which to parse the template.
            from_element (xml.etree.ElementTree.Element): The Element object representing the template.
            type (str): The type of the template, either 'control' or 'treatment' (only considered when no source is provided).
        """

        # If more than one sources are given, raise an error
        sources = [from_string, from_file, from_element]
        if len([source for source in sources if source is not None]) > 1:
            raise ValueError(
                "Only one source can be provided: from_string, from_file, or from_element."
            )

        # Parse the template from the given source
        self._data: ET.Element = None
        if from_string is not None:
            self._data = ET.fromstring(from_string)
            self._validate(allow_incomplete=False)
        elif from_file is not None:
            self._data = ET.parse(from_file).getroot()
            self._validate(allow_incomplete=False)
        elif from_element is not None:
            # Serialize and parse the element to create a deep copy instead of operating on the original element
            self._data = ET.fromstring(ET.tostring(from_element))
            self._validate(allow_incomplete=False)
        else:
            self._data = ET.Element("template")

            # If a type was given (e.g., 'control' or 'treatment'), store it as an attribute of the root template element
            if type is not None:
                self._data.set("type", type)

    def add_situation(self, situation: str) -> None:
        """
        Adds a situation element to the template.

        Args:
            situation (str): The situation to be added to the template.
        """

        situation_element = ET.Element("situation")
        situation_element.text = situation
        self._data.append(situation_element)
        self._validate(allow_incomplete=True)

    def add_prompt(self, prompt: str) -> None:
        """
        Adds a prompt element to the template.

        Args:
            prompt (str): The prompt to be added to the template.
        """

        prompt_element = ET.Element("prompt")
        prompt_element.text = prompt
        self._data.append(prompt_element)
        self._validate(allow_incomplete=True)

    def add_option(self, option: str) -> None:
        """
        Adds an option element to the template.

        Args:
            option (str): The option to be added to the template.
        """

        option_element = ET.Element("option")
        option_element.text = option
        self._data.append(option_element)
        self._validate(allow_incomplete=True)

    def insert(
        self,
        pattern: str = None,
        text: str = None,
        origin: str = None,
        insertion: Insertion = None,
        insertions: list[Insertion] = None,
        trim_full_stop: bool = True,
    ) -> list[Insertion]:
        """
        Inserts text into a blank in the template based on a pattern to be replaced.
        Text can be user-defined (replacing patterns wrapped by '{{' and '}}') or generated by a model (replacing patterns wrapped by '[[' and ']]').
        Users can pass a list of Insertion objects, an Insertion object, or an insertion specified by a pattern, text, and optionally an origin.

        Args:
            pattern (str): The pattern to be replaced in the template.
            text (str): The text to replace the pattern.
            origin (str): The origin of the text, either 'user' or 'model'. If specified, only patterns matching the specified origin will be replaced. Otherwise, all matching patterns will be replaced.
            insertion (Insertion): The Insertion object to insert into the template.
            insertions (list[Insertion]): A list of Insertion objects to be inserted into the template.
            trim_full_stop (bool): Whether or not to remove fullstops, question marks, and exclamation marks (. ? !) at the end of the text.

        Returns:
            list[Insertion]: A list of insertions made into the template.
        """

        # Handle insertions provided in different formats
        if insertions is not None:
            insertions_made = []
            for insertion in insertions:
                insertions_made.extend(self.insert(insertion=insertion))
            return insertions_made
        elif insertion is not None:
            return self.insert(
                pattern=insertion.pattern, text=insertion.text, origin=insertion.origin
            )
        elif pattern is None and text is None:
            raise ValueError(
                "Template.insert: An insertion must be provided. Users can pass an Insertion object, a list of Insertion objects, or an insertion specified by a pattern, text, and optionally an origin."
            )

        # Validate that all input parameters have the correct type
        if not isinstance(pattern, str):
            raise ValueError("Template.insert: The pattern must be a string.")
        if not isinstance(text, str):
            raise ValueError("Template.insert: The text must be a string.")

        # Clean the pattern of any [[...]] and {{...}} and validate that the pattern is consistent with the provided origin
        if pattern.startswith("[[") and pattern.endswith("]]"):
            pattern = pattern.strip("[[").strip("]]")
            if origin is None:
                print(
                    "Template.insert: A pattern was provided that is wrapped in [[...]] but no origin was given. Setting origin to 'model'."
                )
                origin = "model"
            elif origin == "user":
                print(
                    "Template.insert: A pattern was provided that is wrapped in [[...]] but origin is 'user'. Did you mean 'model' instead?"
                )
        elif pattern.startswith("{{") and pattern.endswith("}}"):
            pattern = pattern.strip("{{").strip("}}")
            if origin is None:
                print(
                    "Template.insert: A pattern was provided that is wrapped in {{{{...}}}} but no origin was given. Setting origin to 'user'."
                )
                origin = "user"
            elif origin == "model":
                print(
                    "Template.insert: A pattern was provided that is wrapped in {{{{...}}}} but origin is 'model'. Did you mean 'user' instead?"
                )

        # Trim fullstops at the end of the text, if requested
        if trim_full_stop:
            if text.endswith(".") or text.endswith("?") or text.endswith("!"):
                text = text[:-1]

        # Check that a valid origin is provided
        if origin not in ["user", "model", None]:
            raise ValueError("Origin must be one of 'user', 'model', or None.")

        # If no origin is specified, replace all matching patterns for both origins, 'user' and 'model'
        if origin is None:
            insertions = []
            insertions.extend(self.insert(pattern, text, "user"))
            insertions.extend(self.insert(pattern, text, "model"))
            return insertions

        # Prepare the full pattern and text based on the origin
        pattern_full = "{{" + pattern + "}}" if origin == "user" else "[[" + pattern + "]]"
        text_full = "{{" + text + "}}" if origin == "user" else "[[" + text + "]]"

        # Iterate over all elements in this template
        for elem in self._data:
            # Skip over all elements that are not of type 'situation', 'prompt', or 'option' (especially 'insertion' elements)
            if elem.tag not in ["situation", "prompt", "option"]:
                continue

            # Apply all previous insertions to the element's text
            current_text = self._apply_insertions(
                elem.text, drop_user_brackets=True, drop_model_brackets=True
            )

            # If the element's text still contains unfilled gaps matching the pattern, accept and return the insertion
            if pattern_full in current_text:
                return self._accept_insertion(pattern, text, origin)

        # If the pattern cannot be found, return an empty list
        return []

    def get_gaps(
        self, include_filled: bool = False, include_duplicates: bool = False, origin: str = None
    ) -> list[str]:
        """
        Returns a list of all gaps in this template. Gaps are indicated by either {{...}}, to be filled by the user, or [[...]], to be filled by a model.

        Args:
            include_filled (bool): If True, gaps that are already filled (i.e., values were inserted), are also returned.
            origin (str): One of None, 'user', or 'model'. If an origin is provided, only gaps supposed to be filled from that origin will be returned.

        Returns:
            list[str]: A list of all gaps in this template.
        """

        def find_gaps(text: str) -> list[str]:
            # Define the regex pattern to match [[...]] ('model') or {{...}} ('user')
            pattern = r"(\[\[.*?\]\])|(\{\{.*?\}\})"

            # Find all matches of the pattern in the string
            matches = re.findall(pattern, text)

            # Flatten the list of tuples to get all matched strings. Each element in matches is a tuple, where one of the elements is an empty string
            results = [match[0] if match[0] else match[1] for match in matches]

            if origin is None:
                return results
            elif origin == "user":
                return [match for match in results if match.startswith("{{")]
            elif origin == "model":
                return [match for match in results if match.startswith("[[")]
            else:
                raise ValueError(
                    f"Unknown origin '{origin}'. Must be one of None, 'user', or 'model'."
                )

        # Iterate over all elements in this template and find the gaps
        gaps = []
        for elem in self._data:
            # Skip over all elements that are not of type 'situation', 'prompt', or 'option'
            if elem.tag not in ["situation", "prompt", "option"]:
                continue

            # Find gaps in the element's text. If include_filled=False, insertions are applied to the text first so that gaps with insertions are not detected
            text = elem.text
            if not include_filled:
                text = self._apply_insertions(text)

            gaps.extend(find_gaps(text))

        # Remove duplicates if requested
        if not include_duplicates:
            gaps = list(dict.fromkeys(gaps))

        return gaps

    def get_insertions(self, origin: str = None) -> list[Insertion]:
        """
        Returns a list of insertions made into the template. Each insertion has three attributes 'origin' (either 'user' or 'model'), 'instruction', and the inserted 'text'.

        Args:
            origin (str): One of None, 'model', or 'user'. If an origin is provided, only insertions with that origin will be returned.

        Returns:
            list[Insertion]: A list of Insertion objects.
        """

        insertions = self._data.find("insertions")

        if insertions is None:
            return []

        insertions = self._elements_to_insertions(list(insertions))

        if origin is None:
            return insertions
        else:
            return [insertion for insertion in insertions if insertion.origin == origin]

    def format(
        self,
        insert_headings: bool = True,
        show_type: bool = False,
        drop_user_brackets: bool = True,
        drop_model_brackets: bool = True,
        randomly_flip_options: bool = False,
        shuffle_options: bool = False,
        seed: int = 42,
    ) -> str:
        """
        Formats the template into a string.

        Args:
            insert_headings (bool): Whether to insert headings (Situation, Prompt, Answer Options).
            show_type (bool): Whether to show the type of each element using XML-like tags.
            drop_user_brackets (bool): If True, {{ }} indicating user-made insertions will be removed for every gap with an inserted text.
            drop_model_brackets (bool): If True, [[ ]] indicating model-made insertions will be removed for every gap with an inserted text.
            reverse_options (bool): If True, answer options will be reversed. If False, answer options will not be reversed.
            shuffle_options (bool): If True, answer options will be shuffled randomly using the provided seed. If False, answer options will not be shuffled.
            seed (int): The seed used for randomly shuffling answer options. Ignored, if shuffle_options = False and randomly_flip_options = False.

        Returns:
            str: The formatted string of the template.
        """

        # Validate that the template is complete and not corrupted
        self._validate(allow_incomplete=False)

        # Store the final formatted string in a variable
        formatted = ""

        # Define a function to format an individual element
        def format_element(text: str, type: str) -> str:
            # Fill in the gaps in the element's text according to the insertions made into this template
            text = self._apply_insertions(
                text,
                drop_user_brackets=drop_user_brackets,
                drop_model_brackets=drop_model_brackets,
            )

            if show_type:
                return f"<{type}>{text}</{type}>\n"
            return f"{text}\n"

        # Format all situation elements
        if insert_headings:
            formatted += "Situation:\n"
        for elem in self._data.findall("situation"):
            formatted += format_element(elem.text, elem.tag)

        # Format all prompt elements
        if insert_headings:
            formatted += "\nPrompt:\n"
        for elem in self._data.findall("prompt"):
            formatted += format_element(elem.text, elem.tag)

        # Format all option elements
        if insert_headings:
            formatted += "\nAnswer Options:\n"
        option_counter = 1
        for option in self.get_options(
            randomly_flip_options=randomly_flip_options, shuffle_options=shuffle_options, seed=seed
        )[0]:
            formatted += format_element(f"Option {option_counter}: {option}", "option")
            option_counter += 1

        return formatted

    def get_options(
        self,
        randomly_flip_options: bool = False,
        shuffle_options: bool = False,
        apply_insertions: bool = True,
        seed: int = 42,
    ) -> tuple[list[str], list[int]]:
        """
        Gets the answer options defined in this template and offers functionality to randomly shuffle them.

        Args:
            randomly_flip_options (bool): If True, the answer options will be reversed.
            shuffle_options (bool): If True, the answer options will be randomly shuffled using the provided seed.
            apply_insertions (bool): If True, insertions made into this template will be applied to the answer options.
            seed (int): The seed to used for shuffling the answer options.

        Returns:
            tuple[list[str], list[int]]: Returns two lists, one with the answer option texts and one with the original zero-based position of the answer option.
        """

        # Get all options defined in this template
        options = self._data.findall("option")
        options = [elem.text for elem in options]

        # Create a list of indices for the options with their original position, i.e., [0, 1, 2, ...]
        indices = list(range(len(options)))

        # If requested, reverse the options with probability 0.5
        if randomly_flip_options:
            if random.Random(seed).random() < 0.5:
                indices = indices[::-1]

        # If requested, randomly shuffle the options
        if shuffle_options:
            random.Random(seed).shuffle(indices)
        options = [options[i] for i in indices]

        # If requested, apply all insertions made into this template to the options
        if apply_insertions:
            options = [self._apply_insertions(option) for option in options]

        return options, indices

    def _accept_insertion(self, pattern: str, text: str, origin: str) -> list[Insertion]:
        """
        Stores an insertion into this template.

        Args:
            pattern (str): The pattern replaced by the insertion.
            text (str): The text inserted.
            origin (str): The insertion's origin ('user' or 'model').

        Returns:
            list[Insertion]: The insertion that was stored.
        """

        # If no insertions have been made so far, create a new insertions element in this template
        if self._data.find("insertions") is None:
            self._data.append(ET.Element("insertions"))

        # Append this insertion to the insertions element
        insertion = ET.Element("insertion", attrib={"origin": origin, "instruction": pattern})
        insertion.text = text
        self._data.find("insertions").append(insertion)
        return self._elements_to_insertions([insertion])

    def _validate(self, allow_incomplete: bool = False) -> bool:
        """
        Validates that this template is complete and not corrupted. Raises a ValueError if corruptions have been found. Otherwise, returns True.

        Args:
            allow_incomplete (bool): Whether to allow the template to be incomplete (i.e., still missing situation, prompt, or option elements).

        Returns:
            bool: True if the template is complete and not corrupted.
        """

        for elem in self._data:
            # Validate that all elements have a valid tag
            if elem.tag not in ["situation", "prompt", "option", "insertions"]:
                raise ValueError(
                    f"Templates can only contain elements of type situation, prompt, option, or insertions. Found illegal element of type {elem.tag}."
                )

            # Validate that situation, prompt, and option elements have no further children
            if elem.tag in ["situation", "prompt", "option"] and len(elem) > 0:
                raise ValueError(
                    f"Situation, prompt, and option elements cannot have children. Found a {elem.tag} element with {len(elem)} children."
                )

            # Validate that all situation, prompt, and option elements have text
            if elem.tag in ["situation", "prompt", "option"] and (
                elem.text is None or elem.text.strip() == ""
            ):
                raise ValueError(
                    f'Situation, prompt, and option elements must not be empty. Found a {elem.tag} element with text "{elem.text}"'
                )

        # Validate that all element types appear in sufficient quantity
        if not allow_incomplete:
            if len(self._data.findall("situation")) == 0:
                raise ValueError("The template must contain at least one situation element.")
            if len(self._data.findall("prompt")) != 1:
                raise ValueError(
                    f"The template must contain exactly one prompt element. Found {len(self._data.findall('prompt'))}."
                )
            if len(self._data.findall("option")) < 2:
                raise ValueError(
                    f"The template must contain at least two option elements. Found {len(self._data.findall('option'))}."
                )

        # Validate that situation, prompt, and option elements appear strictly in that order
        last = None
        for elem in self._data:
            if elem.tag == "insertions":
                # Ignore insertion elements as they are invisible to the user
                continue
            if elem.tag == "situation":
                if last not in [None, "situation"]:
                    raise ValueError(
                        f"Situation elements cannot follow {last} elements. Situation elements must be first in a template."
                    )
            elif elem.tag == "prompt":
                if last not in [None, "situation"]:
                    raise ValueError(
                        f"Prompt elements must directly follow situation elements. Found a prompt element following a {last} element."
                    )
            elif elem.tag == "option":
                if last not in [None, "prompt", "option"]:
                    raise ValueError(
                        f"Option elements must directly follow prompt or other option elements. Found an option element following a {last} element."
                    )
            last = elem.tag

        # Validate that insertions have the correct format
        insertions = self._data.findall("insertions")
        if len(insertions) > 1:
            raise ValueError(
                f"There can only be one insertions element in a template. Found {len(insertions)}."
            )
        if len(insertions) == 1:
            for elem in insertions[0]:
                if elem.tag != "insertion":
                    raise ValueError(
                        f"The insertions element must contain only insertion elements. Found a {elem.tag} element inside the insertions element."
                    )
                if "origin" not in elem.attrib:
                    raise ValueError(
                        "Insertion elements must have an origin attribute. Found an insertion element without origin."
                    )
                if elem.attrib["origin"] not in ["user", "model"]:
                    raise ValueError(
                        f"The origin of an insertion element must be either 'user' or 'model'. Found origin {elem.attrib['origin']}."
                    )
                if "instruction" not in elem.attrib:
                    raise ValueError(
                        "Insertion elements must have an instruction attribute. Found an insertion element without instruction."
                    )
                if elem.attrib["instruction"].strip() in [None, ""]:
                    raise ValueError(
                        f"The instruction of an insertion element must not be empty. Found empty instruction '{elem.attrib['instruction']}'."
                    )

        return True

    def _apply_insertions(
        self, text: str, drop_user_brackets: bool = True, drop_model_brackets: bool = True
    ) -> str:
        """
        Applies all insertions that were made into this template to the provided text.

        Args:
            text (str): The text to which to apply the insertions.
            drop_user_brackets (bool): If True, {{ }} indicating user-made insertions will be removed for every gap with an inserted text.
            drop_model_brackets (bool): If True, [[ ]] indicating model-made insertions will be removed for every gap with an inserted text.

        Returns:
            str: The adjusted text where insertions are made into the indicated gaps.
        """

        # Get the insertions that were made into this template
        insertions = self.get_insertions()

        # Iterate over all insertions and apply them to the text
        for insertion in insertions:
            # Expand the search pattern and inserted text with the respective brackets where needed
            if insertion.origin == "user":
                pattern = "{{" + insertion.pattern + "}}"
                if not drop_user_brackets:
                    insertion.text = "{{" + insertion.text + "}}"
            elif insertion.origin == "model":
                pattern = "[[" + insertion.pattern + "]]"
                if not drop_model_brackets:
                    insertion.text = "[[" + insertion.text + "]]"

            # Apply the insertion to the text
            text = text.replace(pattern, insertion.text)

        return text

    def _elements_to_insertions(self, insertions: list[ET.Element]) -> list[Insertion]:
        """
        Converts a list of xml.etree.ElementTree.Element objects representing insertions into a list of Insertion objects.

        Args:
            insertions (list[xml.etree.ElementTree.Element]): A list of Element objects representing insertions.

        Returns:
            list[Insertion]: A list of Insertion objects.
        """

        return [
            Insertion(
                pattern=insertion.attrib["instruction"],
                text=insertion.text,
                origin=insertion.attrib["origin"],
            )
            for insertion in insertions
        ]

    def __str__(self) -> str:
        return self.format(insert_headings=True, show_type=False)

    def __repr__(self) -> str:
        return self.format(insert_headings=False, show_type=True)


class TestConfig:
    """
    A class representing a configuration file for a cognitive bias test.

    Attributes:
        config (xml.etree.ElementTree.ElementTree): An ElementTree object representing the XML configuration file.
    """

    def __init__(self, path: str):
        """
        Instantiates a new TestConfig object.

        Args:
            path (str): The path to the XML configuration file.
        """

        self.config = self._load(path)

    def get_bias_name(self) -> str:
        """
        Returns the name of the cognitive bias being tested.

        Returns:
            str: The name of the cognitive bias being tested.
        """

        return self.config.getroot().get("bias")

    def get_custom_values(self) -> dict:
        """
        Returns the custom values defined in the configuration file.

        Returns:
            dict: A dictionary containing the custom values defined in the configuration file.
        """

        custom_values = self.config.getroot().findall("custom_values")
        custom_values_dict = {}
        for custom_value in custom_values:
            key = custom_value.get("name")
            if len(custom_value) == 0:
                custom_values_dict[key] = None
            elif len(custom_value) == 1:
                custom_values_dict[key] = custom_value.find("value").text
            else:
                custom_values_dict[key] = [value.text for value in custom_value]

        return custom_values_dict

    def get_variants(self) -> list[str]:
        """
        Returns a list of variant names defined in the configuration file.

        Returns:
            A list of variant names defined in the configuration file.
        """

        # Find all variant elements in the configuration file
        variants = self.config.getroot().findall("variant")

        # Return the names of all variants
        return [variant.get("name") for variant in variants]

    def get_template(self, template_type: str = "control", variant: str = None) -> Template:
        """
        Returns a template from the test configuration.

        Args:
            template_type (str): The type of the template ('control' or 'treatment').
            variant (str): The name of the variant. Only needed if the test configuration includes multiple variants.

        Returns:
            Template: A Template object representing the template.
        """

        root = self.config.getroot()

        if variant is not None:
            # Find the variant element with the specified name
            variant_element = root.find(f"variant[@name='{variant}']")
            if variant_element is None:
                raise ValueError(f"Variant '{variant}' not found in the configuration.")
        else:
            # No variant was specified, try to find the next best variant
            found_variants = root.findall("variant")
            if len(found_variants) > 1:
                raise ValueError(
                    f"{len(found_variants)} variants found in the configuration. Please specify in which variant to find the template."
                )
            elif len(found_variants) == 1:
                variant_element = found_variants[0]
            else:
                # No variant elements in the configuration file, treat the root as the variant element
                variant_element = root

        # Find the template with the specified type
        template_config = variant_element.find(f"template[@type='{template_type}']")
        if template_config is None:
            raise ValueError(
                f"No template with type '{template_type}' found in variant '{variant}'."
            )

        # Parse the template
        template = Template(from_element=template_config)

        return template

    def get_control_template(self, variant: str = None) -> Template:
        """
        Returns the control template for the specified variant or the Default variant if none is specified.

        Args:
            variant (str): The name of the variant.

        Returns:
            Template: A Template object representing the control template.
        """

        return self.get_template("control", variant)

    def get_treatment_template(self, variant: str = None) -> Template:
        """
        Returns the treatment template for the specified variant or the Default variant if none is specified.

        Args:
            variant (str): The name of the variant.

        Returns:
            Template: A Template object representing the treatment template.
        """

        return self.get_template("treatment", variant)

    def _load(self, path: str) -> ET.ElementTree:
        """
        Loads the XML configuration file for the specified cognitive bias.

        Args:
            path (str): The path to the XML configuration file.

        Returns:
            An xml.etree.ElementTree.ElementTree object representing the XML configuration file.
        """

        return ET.parse(path)


class TestCase:
    """
    A class representing a cognitive bias test case.

    Attributes:
        BIAS (str): The name of the cognitive bias being tested.
        CONTROL (Template): The control template for the test case.
        TREATMENT (Template): The treatment template for the test case.
        GENERATOR (str): The name of the LLM generator used to populate the templates.
        SEED (int): The seed used to generate the test case.
        TEMPERATURE (float): The temperature used for sampling from the LLM generator.
        TIMESTAMP (str): The timestamp when the test case was created.
        SCENARIO (str): The scenario in which the test case is being conducted.
        VARIANT (str, optional): The variant of the test case.
        REMARKS (str, optional): Any additional remarks about the test case.
    """

    def __init__(
        self,
        id: int,
        bias: str,
        condition: str,
        template: Template,
        generator: str,
        temperature: float,
        seed: int,
        scenario: str,
        variant: str = None,
        remarks: str = None,
        **kwargs,
    ):
        self.ID: int = id
        self.BIAS: str = bias
        self.CONDITION: Template = condition
        self.TEMPLATE: Template = template
        self.GENERATOR: str = generator
        self.SEED: int = seed
        self.TEMPERATURE: float = temperature
        self.TIMESTAMP: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.SCENARIO: str = scenario
        self.VARIANT: str = variant
        self.REMARKS: str = remarks

        # Add additional fields that are passed but not natively supported
        for key, value in kwargs.items():
            setattr(self, key.upper(), value)

    def __str__(self) -> str:
        return f"---TestCase---\n\nBIAS: {self.BIAS}\nVARIANT: {self.VARIANT}\nSCENARIO: {self.SCENARIO}\nGENERATOR: {self.GENERATOR}\nTEMPERATURE: {self.TEMPERATURE}\nTIMESTAMP: {self.TIMESTAMP}\nSEED: {self.SEED}\n\TEMPLATE:\n{self.TEMPLATE}\n\nREMARKS:\n{self.REMARKS}\n\n------"

    def __repr__(self) -> str:
        return self.__str__()
