# Konvoluciona neuralna mreža za učenje igre Breakout

Implementacija Deepmind DQN rada iz 2015 godine za igru Breakout. 

## Preuzimanje koda

```sh
# Preuzimanje izvornog koda
git clone https://github.com/dulex123/diplomski

# Instalacija neophodnih paketa
sudo apt-get install python3-pip
sudo pip3 install numpy tensorflow numpy scikit-image gym[atari]
```

## Pokretanje

```sh
# Treniranje mreže na podrazumevanom okruženju (Breakout-v0)
python3 main.py

# Pokretanje mreže sa istreniranim težinama i prikazom igre
# težine se podrazumevano nalaze u saved_networks/<naziv_okruženja>
python3 main.py --evaluate --render
```

# 2017

Released under the [MIT License].<br>
Authored and maintained by Dušan Josipović.

> Blog [dulex123.github.io](http://dulex123.github.io) &nbsp;&middot;&nbsp;
> GitHub [@dulex123](https://github.com/dulex123) &nbsp;&middot;&nbsp;
> Twitter [@josipovicd](https://twitter.com/josipovicd)

[MIT License]: http://mit-license.org/
