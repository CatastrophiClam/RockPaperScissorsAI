package challenges;
import java.util.*;


public class RockPaperScissorsPlayer {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner scan = new Scanner(System.in);
		RockPaperScissorsAI ai = new RockPaperScissorsAI();
		int input;
		int wins = 0;
		int losses = 0;
		int ties = 0;
		int answer;
		String[] key = new String[]{"","Rock","Paper","Scissors"};
		
		while (true){
			System.out.println("1 - Rock,  2 - Paper,  3 - Scissors,  4 - Stats: ");
			input = scan.nextInt();
			
			if (input == 4){
				System.out.println("Wins: "+wins+" Losses: "+losses+" Ties: "+ties+" Win Percentage: "+(wins/(wins+losses+ties+0.0)*100)+"%");
			}else if (input == 5){
				double[][] biases = ai.getBiases();
				for (int i = 0; i < biases.length; i++){
					System.out.print("[");
					for (int j = 0; j < biases[i].length; j++){
						System.out.printf("%.2f, ",biases[i][j]);
					}
					System.out.println("]");
				}
			}else{
				answer = ai.next();
				System.out.println(key[answer]);
				if (input-answer == 1 || input-answer == -2){
					System.out.println("You win!");
					wins++;
				}else if(input - answer == 0){
					System.out.println("Tie");
					ties++;
				}else{
					System.out.println("You lose.");
					losses++;
				}
				ai.update(input);
			}
		}
	}

}
