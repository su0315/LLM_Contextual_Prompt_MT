larger_tokens = [
    {'Aber': ['lexical_cohesion']}, {'wenn': []}, {'man': []}, {'ein': []}, {'Außerirdischer': ['lexical_cohesion']}, {'wäre': []}, {',': []}, {'der': []}, {'von': []}, {'all': []}, {'dem': []}, {'nichts': []}, {'weiß': []}, {',': []}, {'der': []}, {'keine': []}, {'Ahnung': []}, {'von': []}, {'irdischer': []}, {'Intelligenz': ['lexical_cohesion']}, {'hat': []}, {',': []}, {'wäre': []}, {'man': []}, {',': []}, {'eine': []}, {'physikalische': []}, {'Theorie': []}, {'aufzustellen': []}, {',': []}, {'warum': []}, {'Asteroiden': []}, {',': []}, {'die': []}, {'bis': []}, {'zu': []}, {'einem': []}, {'bestimmten': []}, {'Zeitpunkt': []}, {'die': []}, {'Oberfläche': []}, {'des': []}, {'Planeten': []}, {'zerstört': []}, {'haben': []}, {',': []}, {'dies': []}, {'mysteriöserweise': ['lexical_cohesion']}, {'nicht': []}, {'mehr': []}, {'tun': []}, {'.': []}
]

smaller_tokens = [
    {'A': -0.50634766}, {'ber': -0.00032758713}, {'wenn': -0.004169464}, {'man': -4.0664062}, {'ein': -3.4667969}, {'Außer': -1.7519531}, {'ird': -2.7775764}, {'ischer': -0.18054199}, {'w': -2.1621094}, {'äre': -0.051849365}, {',': -0.008453369}, {'der': -0.01574707}, {'von': -0.43115234}, {'all': -0.32373047}, {'dem': -0.6455078}, {'nicht': -0.045074463}, {'s': -0.0006828308}, {'we': -0.91064453}, {'iß': -7.915497}, {',': -0.98583984}, {'der': -0.87158203}, {'keine': -1.1494141}, {'A': -2.9472656}, {'hn': -0.00021493435},{'ung': -5.4121017}, {'von': -0.052612305}, {'ird': -1.6455078}, {'ischer': -2.0503998}, {'Int': -0.3071289}, {'ell': -0.00037193298}, {'igen': -0.00025177002}, {'z': -4.529953}, {'hat': -2.7775764}, {',': -0.046539307}, {'w': -0.0023918152}, {'äre': -1.3027344}, {'man': -0.042114258}, {'ge': -0.0035591125}, {'zw': -0.09442139}, {'ungen': -1.7046928}, {',': -0.00048279762}, {'eine': -0.0048103333}, {'phys': -0.004711151}, {'ikal': -0.013137817}, {'ische': -0.002336502}, {'The': -7.843971}, {'orie': -0.00010359287}, {'auf': -0.00016415119},{'z': -3.9003906}, {'ust': -0.0031719208}, {'ellen': -0.030410767}, {',': -7.83205}, {'war': -0.00049877167}, {'um': -14.140625}, {'Ast': -0.0014810562}, {'ero': -0.890625}, {'iden': -2.1100044}, {',': -0.00026106834}, {'die': -0.66308594}, {'bis': -0.04208374}, {'zu': -0.7192383}, {'einem': -0.0020332336}, {'best': -0.00096797943}, {'imm': -0.18066406}, {'ten': -3.2663345}, {'Zeit': -0.0007519722}, {'punkt': -0.32666016}, {'die': -0.00010538101}, {'Ober': -0.17858887}, {'fl': -0.006465912}, {'äche': -8.702278}, {'des': -0.00026226044}, {'Plan': -7.875}, {'eten': -0.0026416779}, {'z': -2.360344}, {'erst': -0.19592285}, {'ört': -0.011062622}, {'haben': -1.5722656}, {',': -1.0761719}, {'dies': -0.28515625}, {'myster': -2.6289062}, {'i': -1.4384766}, {'ö': -6.54459}, {'ser': -1.5263672}, {'weise': -0.23864746}, {'nicht': -0.0035247803}, {'mehr': -1.0380859}, {'tun': -0.14135742}, {'.': -0.0045814514}
]


def is_substring(larger_str, smaller_str):
    return smaller_str in larger_str

def take_continuous_correlation(larger_str, smaller_tokens):
    current_continuous_correlation = []
    original_larger_str = larger_str
    for smaller_item in smaller_tokens:
        for smaller_key, smaller_value in smaller_item.items():
            smaller_str = smaller_key  # Convert to lowercase for case-insensitive comparison            
            if is_substring(larger_str, smaller_str):
                current_continuous_correlation.append({smaller_key: smaller_value})
                larger_str = larger_str[len(smaller_str):]  # Remove matched substring from larger_str
                if current_continuous_correlation and larger_str == "":
                    return current_continuous_correlation
            else:
                larger_str = original_larger_str  # Reset to the original larger_str
                if current_continuous_correlation and larger_str !="": # if couldn_t complete the larger_str correlation, remove the partial correlation
                    current_continuous_correlation.remove(current_continuous_correlation[-1])
                
    return current_continuous_correlation if not larger_str else []

def extract_continuous_correlations(larger_tokens, smaller_tokens):
    relevant_items = []
    current_continuous_correlation = []

    for item in larger_tokens:
        for key, value in item.items():
            if value and "lexical_cohesion" in value:
                larger_str = key  # Convert to lowercase for case-insensitive comparison
                continuous_correlation = take_continuous_correlation(larger_str, smaller_tokens) ###here
                if continuous_correlation:
                    if current_continuous_correlation:  # Check if there's a previous correlation
                        relevant_items.append(current_continuous_correlation)
                        current_continuous_correlation = []  # Reset for the next correlation
                    relevant_items.append(continuous_correlation)
                    break  # Exit loop if a relevant item is found
                else:
                    current_continuous_correlation.extend(continuous_correlation)

    if current_continuous_correlation:  # Check if there's a continuous correlation at the end
        relevant_items.append(current_continuous_correlation)

    return relevant_items

result = extract_continuous_correlations(larger_tokens, smaller_tokens)
print(result)




