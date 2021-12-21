# Glass House 
This is repository where I dump my implementation of various **reinforcement learing algortihms**. Each implementaiton is rewritten using **pytorch** and exists in one file <algorithm name>.py . These are ment to be toy examples, are primarily for learning and expirmentation.

## Usage
You can train the model by executing the following command:
```bash
python <algorithm name>.py 
```
 
Implementation of the algorithms are benchmarked againts Openai gym the cart pole problem.
<p align="center">
	<img src="res/cartpole.gif" width="200" /> 
</p>
	
### Algorithm

<table>
  <tr>
    <td> <img src="res/DQN.png" width="300"/> </td>
    <td> <img src="res/PPO.png" width="300"/> </td>
   </tr> 
   <tr>
      <td> DQN: Deep Q-learning </td>
      <td> PPO: Proximal policy optimization </td>
  </tr>
</table>


## Dependencies
- PyTorch
- OpenAI GYM

## Reference 
- [Youtube-Code-Repository](https://github.com/philtabor/Youtube-Code-Repository)
- [minimalRL-pytorch] (https://github.com/seungeunrho/minimalRL)
