from pathlib import Path

import pandas

if __name__ == '__main__':
    """
    In this example, we search for positive transitions of Steer_Warning, which indicate that the EPS
    has stopped responding to our messages. This analysis would allow you to find the cause of these
    steer warnings and potentially work around them.
    """

    cartella_csv = Path("../SUBARU_OUTBACK")

    lista_df = []

    for file in cartella_csv.glob("*.csv"):
        try:
            df = pandas.read_csv(file)
            if sum(df["Steer_Error_1"]) > 0:
                lista_df.append(df)
            print(f"Letto: {file.name} ({len(df)} righe)")
        except Exception as e:
            print(f"Errore con {file.name}: {e}")

    perc = 100
    split_i = int(perc/100*len(lista_df))

    df_unificato = pandas.concat(lista_df[0:split_i], ignore_index=True)
    df_unificato = df_unificato.sort_values('time')
    print(f"\nDataFrame train: {df_unificato.shape}")
    stats = df_unificato.describe()
    df_unificato.to_csv("output_steererr_" + str(perc) + "_train.csv", index=False)
    stats.to_csv("output_steererr_" + str(perc) + "_train_stats.csv")

    df_unificato = pandas.concat(lista_df[split_i:], ignore_index=True)
    df_unificato = df_unificato.sort_values('time')
    print(f"\nDataFrame test: {df_unificato.shape}")
    stats = df_unificato.describe()
    df_unificato.to_csv("output_steererr_" + str(perc) + "_test.csv", index=False)
    stats.to_csv("output_steererr_" + str(perc) + "_test_stats.csv")
